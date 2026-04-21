from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import SPLITS_DIR
from src.data.preprocess_image import preprocess_image, to_grayscale
from src.features.hog_features import extract_hog_features


def build_hog_features_from_split(csv_path: Path, image_size=(128, 128)):
    df = pd.read_csv(csv_path)

    X = []
    y = []

    for idx, row in df.iterrows():
        image = preprocess_image(row["image_path"], size=image_size)
        gray = to_grayscale(image)
        features = extract_hog_features(gray)

        X.append(features)
        y.append(row["label"])

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(df)} from {csv_path.name}")

    return np.array(X), np.array(y), df


def train_logistic_regression(X_train, y_train):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            multi_class="multinomial",
            solver="lbfgs",
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X, y, split_name="Validation"):
    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, average="macro")

    print(f"\n{split_name} Accuracy: {acc:.4f}")
    print(f"{split_name} Macro F1: {macro_f1:.4f}")

    print(f"\n{split_name} Classification Report:")
    print(classification_report(y, y_pred))

    print(f"{split_name} Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "y_pred": y_pred,
    }


if __name__ == "__main__":
    train_path = SPLITS_DIR / "train.csv"
    val_path = SPLITS_DIR / "val.csv"
    test_path = SPLITS_DIR / "test.csv"

    print("Extracting HOG features for train split...")
    X_train, y_train, train_df = build_hog_features_from_split(train_path)

    print("\nExtracting HOG features for validation split...")
    X_val, y_val, val_df = build_hog_features_from_split(val_path)

    print("\nExtracting HOG features for test split...")
    X_test, y_test, test_df = build_hog_features_from_split(test_path)

    print("\nTraining Logistic Regression model...")
    model = train_logistic_regression(X_train, y_train)

    evaluate_model(model, X_val, y_val, split_name="Validation")
    evaluate_model(model, X_test, y_test, split_name="Test")