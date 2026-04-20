# src/features/tfidf_features.py

from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer(
    max_features: int = 10000,
    ngram_range=(1, 2),
    min_df: int = 2,
    max_df: float = 0.9
):
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        stop_words="english"
    )


def fit_transform_tfidf(train_texts, val_texts=None, test_texts=None, **kwargs):
    vectorizer = build_tfidf_vectorizer(**kwargs)

    X_train = vectorizer.fit_transform(train_texts)

    outputs = {
        "vectorizer": vectorizer,
        "X_train": X_train
    }

    if val_texts is not None:
        outputs["X_val"] = vectorizer.transform(val_texts)

    if test_texts is not None:
        outputs["X_test"] = vectorizer.transform(test_texts)

    return outputs
