# src/models/text_models.py

import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression


def build_tfidf_baseline_model(C: float = 1.0, max_iter: int = 1000):
    base_model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="liblinear"
    )
    model = OneVsRestClassifier(base_model)
    return model


def train_text_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def predict_text_model(model, X):
    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)
    else:
        y_prob = None

    return y_pred, y_prob
