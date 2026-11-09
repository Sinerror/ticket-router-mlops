"""pyfunc wrapper so a registered version is also loadable with plain `mlflow.pyfunc.load_model`."""

from __future__ import annotations

import mlflow


class TicketRouterModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from ticketflow.inference import OnnxPredictor

        self.predictor = OnnxPredictor(context.artifacts["bundle"], variant="int8")

    def predict(self, context, model_input, params=None):
        texts = list(model_input["text"]) if hasattr(model_input, "columns") else list(model_input)
        return [p.label for p in self.predictor.predict_batch(texts)]
