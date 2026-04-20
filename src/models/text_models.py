from sklearn.linear_model import LogisticRegression


def build_tfidf_baseline_model(C=1.0, max_iter=1000):
    return LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="liblinear",
        multi_class="ovr"
    )


def build_bert_classifier(C=1.0, max_iter=1000):
    return LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="liblinear",
        multi_class="ovr"
    )


def train_text_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def predict_text_model(model, X):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    return y_pred, y_prob
