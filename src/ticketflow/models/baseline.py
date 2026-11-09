"""Classical baseline: TF-IDF (words + char n-grams) + logistic regression."""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


def build_baseline(seed: int = 42) -> Pipeline:
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
        ]
    )
    clf = LogisticRegression(C=10.0, max_iter=2000, random_state=seed)
    return Pipeline([("features", features), ("clf", clf)])


def train_baseline(texts: Sequence[str], labels: Sequence[str], seed: int = 42) -> Pipeline:
    model = build_baseline(seed)
    model.fit(list(texts), list(labels))
    return model
