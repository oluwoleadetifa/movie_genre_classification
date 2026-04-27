from pathlib import Path
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


PROJECT_DIR = Path("/content/movie_genre_classification")
OUTPUT_DIR = PROJECT_DIR / "outputs"
PROBS_DIR = OUTPUT_DIR / "probs"
FEATURES_DIR = OUTPUT_DIR / "features"
METRICS_DIR = OUTPUT_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["action", "comedy", "horror", "romance"]


def evaluate(name, y_true, preds):
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_true, preds)),
        "macro_f1": float(f1_score(y_true, preds, average="macro")),
        "classification_report": classification_report(y_true, preds, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, preds).tolist(),
    }


def main():
    bert_test_probs = np.load(PROBS_DIR / "bert_test_probs.npy")
    resnet_test_probs = np.load(PROBS_DIR / "resnet_test_probs.npy")
    y_test = np.load(PROBS_DIR / "y_test.npy", allow_pickle=True)

    if bert_test_probs.shape != resnet_test_probs.shape:
        raise ValueError(f"Shape mismatch: BERT {bert_test_probs.shape}, ResNet {resnet_test_probs.shape}")

    results = []

    print("Late fusion results")
    for w_bert in [0.5, 0.6, 0.7, 0.8, 0.9]:
        w_resnet = 1.0 - w_bert
        fused_probs = (w_bert * bert_test_probs) + (w_resnet * resnet_test_probs)
        fused_preds = fused_probs.argmax(axis=1)

        # If labels are strings, convert numeric argmax back to class names.
        if y_test.dtype.kind in {"U", "S", "O"}:
            fused_preds = np.array([CLASS_NAMES[i] for i in fused_preds])

        row = evaluate(f"Late Fusion BERT {w_bert:.1f} / ResNet {w_resnet:.1f}", y_test, fused_preds)
        results.append(row)
        print(f"{row['model']}: acc={row['accuracy']:.4f}, macro_f1={row['macro_f1']:.4f}")

    print("\nIntermediate fusion results")
    X_train_bert = np.load(FEATURES_DIR / "bert_train_embeddings.npy")
    X_test_bert = np.load(FEATURES_DIR / "bert_test_embeddings.npy")
    X_train_resnet = np.load(FEATURES_DIR / "resnet_train_features.npy")
    X_test_resnet = np.load(FEATURES_DIR / "resnet_test_features.npy")
    y_train = np.load(PROBS_DIR / "y_train.npy", allow_pickle=True)

    X_train_fused = np.concatenate([X_train_bert, X_train_resnet], axis=1)
    X_test_fused = np.concatenate([X_test_bert, X_test_resnet], axis=1)

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced")
    )
    clf.fit(X_train_fused, y_train)
    intermediate_preds = clf.predict(X_test_fused)

    row = evaluate("Intermediate Fusion BERT Embeddings + ResNet Features", y_test, intermediate_preds)
    results.append(row)
    print(f"{row['model']}: acc={row['accuracy']:.4f}, macro_f1={row['macro_f1']:.4f}")

    with open(METRICS_DIR / "week3_fusion_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved:", METRICS_DIR / "week3_fusion_results.json")


if __name__ == "__main__":
    main()
