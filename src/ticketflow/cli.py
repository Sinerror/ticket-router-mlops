"""Command line entry point: `ticketflow <command>` (or `python -m ticketflow <command>`)."""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def _print(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def cmd_download(a):
    from ticketflow.data import download_raw

    print(download_raw(force=a.force))


def cmd_prepare(_):
    from ticketflow.data import prepare

    _print(prepare())


def cmd_train(a):
    """Train + gate + register. Without --upto-day trains the bootstrap model and promotes it."""
    from ticketflow import registry
    from ticketflow.pipeline import gate_and_register, make_datasets, train_candidate

    ds = make_datasets(a.upto_day)
    name = "bootstrap" if a.upto_day is None else f"retrain_day{a.upto_day}"
    cand = train_candidate(ds, name, tags={"trigger": a.trigger})
    registry.setup()
    prod_bundle = None
    if a.upto_day is not None and registry.get_alias_version(registry.PROD):
        prod_bundle = registry.download_bundle(registry.PROD, cand.bundle_dir.parent / "_production")
    gate, version = gate_and_register(cand, ds.golden, prod_bundle)
    if version and (a.upto_day is None or a.promote):
        registry.promote(version)
    _print(
        {
            "run_id": cand.run_id,
            "metrics": {k: cand.metrics[k] for k in ("macro_f1", "accuracy")},
            "baseline_macro_f1": cand.baseline_metrics["macro_f1"],
            "gate": gate.to_dict(),
            "version": version,
        }
    )
    return 0 if gate.passed else 3


def cmd_simulate(a):
    from ticketflow.simulate import run

    s = run(gate_drill=not a.no_gate_drill, rollback_drill=True)
    _print({k: v for k, v in s.items() if k != "events"})


def cmd_benchmark(a):
    from ticketflow.benchmark import run
    from ticketflow.config import ARTIFACTS_DIR
    from ticketflow.data import load_stream

    stream = load_stream()
    if a.bundle:
        bundle = a.bundle
    else:  # default: the first model retrained after the drift in `ticketflow simulate`
        found = sorted(
            (ARTIFACTS_DIR / "simulation" / "candidates").glob("retrain_day*"),
            key=lambda d: int(d.name.removeprefix("retrain_day")),
        )
        if not found:
            raise SystemExit("run `ticketflow simulate` first or pass --bundle")
        bundle = str(found[0])
    eval_df = stream[stream["month"].between(a.from_month, a.to_month)]
    _print(run(bundle, eval_df, n_latency=a.n, threads=a.threads))


def cmd_promote(a):
    from ticketflow import registry

    registry.setup()
    registry.promote(a.version)
    _print({"production": a.version})


def cmd_rollback(a):
    """production <- previous; with --reload-url also makes the service switch and times the whole thing."""
    import time

    from ticketflow import registry

    t0 = time.perf_counter()
    registry.setup()
    frm, to = registry.rollback()
    out = {"rolled_back_from": frm, "production": to}
    if a.reload_url:
        from ticketflow.rollout import ServiceClient

        client = ServiceClient(url=a.reload_url, token=a.token)
        out["service"] = client.reload()
        assert client.health()["production"] == to, "service did not switch"
    out["seconds_to_100pct_traffic"] = round(time.perf_counter() - t0, 2)
    _print(out)


def cmd_replay(a):
    """Send the simulated stream to the running service, day by day, plus operator feedback."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    import requests

    from ticketflow.config import DATA_DIR
    from ticketflow.data import load_stream

    stream = load_stream()
    state = DATA_DIR / "state"
    state.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    def send(row):
        r = session.post(
            f"{a.url}/predict", json={"ticket_id": str(row.ticket_id), "text": row.text}, timeout=10
        )
        r.raise_for_status()
        if row.labelled:
            session.post(
                f"{a.url}/feedback", json={"ticket_id": str(row.ticket_id), "label": row.label}, timeout=10
            )
        return r.json()

    last = a.start_day + a.days if a.days else int(stream["day"].max()) + 1
    with ThreadPoolExecutor(4) as pool:
        for day in range(a.start_day, last):
            t0 = time.time()
            x = stream[stream["day"] == day]
            res = list(pool.map(send, x.itertuples()))
            acc = sum(r["label"] == lab for r, lab in zip(res, x["label"], strict=True)) / max(1, len(x))
            versions = sorted({r["model_version"] for r in res})
            (state / "current_day.json").write_text(json.dumps({"day": day}), "utf-8")
            print(f"day {day:3d}: {len(x)} tickets, accuracy {acc:.3f}, versions {versions}", flush=True)
            time.sleep(max(0.0, a.seconds_per_day - (time.time() - t0)))


def cmd_drift_check(a):
    from ticketflow.monitor import run_drift_check

    _print(run_drift_check(a.log))


def cmd_serve(a):
    import uvicorn

    uvicorn.run("ticketflow.serving.app:app", host=a.host, port=a.port)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ticketflow")
    sub = ap.add_subparsers(dest="cmd", required=True)
    dl = sub.add_parser("download", help="fetch the raw dataset from the Hugging Face Hub")
    dl.add_argument("--force", action="store_true")
    dl.set_defaults(fn=cmd_download)
    sub.add_parser("prepare", help="download dataset, build history + month stream").set_defaults(
        fn=cmd_prepare
    )
    t = sub.add_parser("train", help="train candidate, apply quality gate, register in MLflow")
    t.add_argument("--upto-day", type=int, default=None)
    t.add_argument("--trigger", default="manual")
    t.add_argument("--promote", action="store_true", help="promote right away (skip shadow/canary)")
    t.set_defaults(fn=cmd_train)
    s = sub.add_parser("simulate", help="replay six months with drift through the full lifecycle")
    s.add_argument("--no-gate-drill", action="store_true")
    s.set_defaults(fn=cmd_simulate)
    b = sub.add_parser("benchmark", help="PyTorch vs ONNX vs ONNX int8 latency/quality on CPU")
    b.add_argument("--bundle", default=None)
    b.add_argument("--from-month", type=int, default=4)
    b.add_argument(
        "--to-month", type=int, default=5, help="default: post-drift months the retrained model serves"
    )
    b.add_argument("--n", type=int, default=300)
    b.add_argument("--threads", type=int, default=1)
    b.set_defaults(fn=cmd_benchmark)
    pr = sub.add_parser("promote", help="move the production alias to a version")
    pr.add_argument("version")
    pr.set_defaults(fn=cmd_promote)
    rb = sub.add_parser("rollback", help="production <- previous")
    rb.add_argument("--reload-url", default=None, help="service URL to reload after moving the alias")
    rb.add_argument("--token", default=os.environ.get("ADMIN_TOKEN"))
    rb.set_defaults(fn=cmd_rollback)
    rp = sub.add_parser("replay", help="replay the ticket stream against the running service")
    rp.add_argument("--url", default=os.environ.get("SERVICE_URL", "http://localhost:8000"))
    rp.add_argument("--start-day", type=int, default=0)
    rp.add_argument("--days", type=int, default=0, help="0 = until the end of the stream")
    rp.add_argument("--seconds-per-day", type=float, default=10.0)
    rp.set_defaults(fn=cmd_replay)
    dc = sub.add_parser("drift-check", help="Evidently drift check over the service prediction log")
    dc.add_argument("--log", default=os.environ.get("PREDICTION_LOG", "data/state/predictions.jsonl"))
    dc.set_defaults(fn=cmd_drift_check)
    sv = sub.add_parser("serve", help="run the FastAPI service")
    sv.add_argument("--host", default="0.0.0.0")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(fn=cmd_serve)
    a = ap.parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
