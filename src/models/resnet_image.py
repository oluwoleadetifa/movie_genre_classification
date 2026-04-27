from pathlib import Path
import copy
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from src.config import SPLITS_DIR, OUTPUT_DIR
from src.data.image_dataset import MoviePosterDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_CLASSES = 4

# Keep this order consistent with the text model and fusion notebook.
CLASS_NAMES = ["action", "comedy", "horror", "romance"]


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return train_transform, eval_transform


def get_dataloaders():
    train_transform, eval_transform = get_transforms()

    train_dataset = MoviePosterDataset(SPLITS_DIR / "train.csv", transform=train_transform)
    val_dataset = MoviePosterDataset(SPLITS_DIR / "val.csv", transform=eval_transform)
    test_dataset = MoviePosterDataset(SPLITS_DIR / "test.csv", transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Separate non-augmented training loader for saving train probabilities/features.
    train_eval_dataset = MoviePosterDataset(SPLITS_DIR / "train.csv", transform=eval_transform)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, train_eval_loader


def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    return model.to(DEVICE)


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)

            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, acc, macro_f1, np.array(all_labels), np.array(all_preds)


@torch.no_grad()
def collect_probs_and_features(model, loader):
    """Return probabilities, ResNet penultimate-layer features, labels, and predictions."""
    model.eval()

    feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(DEVICE)
    feature_extractor.eval()

    all_probs = []
    all_features = []
    all_labels = []
    all_preds = []

    for images, labels in loader:
        images = images.to(DEVICE)

        logits = model(images)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        features = feature_extractor(images)
        features = torch.flatten(features, start_dim=1)

        all_probs.append(probs.cpu().numpy())
        all_features.append(features.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

    return (
        np.vstack(all_probs),
        np.vstack(all_features),
        np.array(all_labels),
        np.array(all_preds),
    )


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def train_model():
    train_loader, val_loader, test_loader, train_eval_loader = get_dataloaders()
    model = build_model()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_f1 = -1.0
    best_model_wts = copy.deepcopy(model.state_dict())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_dir = OUTPUT_DIR / "models"
    metrics_dir = OUTPUT_DIR / "metrics"
    probs_dir = OUTPUT_DIR / "probs"
    features_dir = OUTPUT_DIR / "features"

    for d in [model_dir, metrics_dir, probs_dir, features_dir]:
        d.mkdir(parents=True, exist_ok=True)

    history = []

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc, train_f1, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_f1, _, _ = run_epoch(model, val_loader, criterion)

        epoch_row = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "train_macro_f1": float(train_f1),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc),
            "val_macro_f1": float(val_f1),
        }
        history.append(epoch_row)

        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train Macro F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} | Val   Macro F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), model_dir / "best_resnet18.pth")
            print("Saved new best model.")

    model.load_state_dict(best_model_wts)

    test_loss, test_acc, test_f1, test_labels, test_preds = run_epoch(model, test_loader, criterion)

    print("\nBest Validation Macro F1:", f"{best_val_f1:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Macro F1: {test_f1:.4f}")

    print("\nTest Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES))

    print("Test Confusion Matrix:")
    print(confusion_matrix(test_labels, test_preds))

    # Save metrics.
    results_resnet = {
        "best_val_macro_f1": float(best_val_f1),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_f1),
        "class_names": CLASS_NAMES,
        "classification_report": classification_report(
            test_labels,
            test_preds,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(test_labels, test_preds).tolist(),
        "history": history,
    }
    save_json(metrics_dir / "resnet_results.json", results_resnet)
    save_json(OUTPUT_DIR / "class_order_resnet.json", {"class_names": CLASS_NAMES})

    # Save probabilities and features for Week 3.
    split_loaders = {
        "train": train_eval_loader,
        "val": val_loader,
        "test": test_loader,
    }

    for split_name, loader in split_loaders.items():
        probs, features, labels, preds = collect_probs_and_features(model, loader)

        np.save(probs_dir / f"resnet_{split_name}_probs.npy", probs)
        np.save(features_dir / f"resnet_{split_name}_features.npy", features)
        np.save(probs_dir / f"y_{split_name}_resnet.npy", labels)
        np.save(probs_dir / f"resnet_{split_name}_preds.npy", preds)

        print(
            f"Saved {split_name}: probs={probs.shape}, "
            f"features={features.shape}, labels={labels.shape}"
        )
        
    # Always save final model
    torch.save(model.state_dict(), model_dir / "final_resnet18.pth")
    print("Saved final model.")

    print("\nDone. ResNet model, metrics, probabilities, and features saved under outputs/.")


if __name__ == "__main__":
    print("Using device:", DEVICE)
    train_model()