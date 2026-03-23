import numpy as np
import torch
from torch.utils.data import DataLoader
import csv
import os

from utils.dataset import IDX_TO_CAT, SUBJECT_IDS


def compute_confusion_matrix(preds, labels, num_classes=20):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for p, l in zip(preds, labels):
        cm[l, p] += 1
    return cm


def print_confusion_analysis(cm, top_k=5):
    num_classes = cm.shape[0]
    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)
    confusions = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm_off[i, j] > 0:
                confusions.append((IDX_TO_CAT[i], IDX_TO_CAT[j], cm_off[i, j]))
    confusions.sort(key=lambda x: x[2], reverse=True)
    print(f"\nTop {top_k} Most Confused Pairs:")
    print("-" * 50)
    for true_cat, pred_cat, count in confusions[:top_k]:
        print(f"  True: {true_cat:15s} -> Predicted: {pred_cat:15s} | Count: {count}")


def print_per_subject_accuracy(per_subject_acc):
    print("\nPer-Subject Accuracy:")
    print("-" * 40)
    for sid in sorted(per_subject_acc.keys()):
        sub_name = SUBJECT_IDS[sid] if sid < len(SUBJECT_IDS) else f"sub-{sid}"
        print(f"  {sub_name}: {per_subject_acc[sid]:.2f}%")
    mean_acc = np.mean(list(per_subject_acc.values()))
    print(f"  {'Mean':>7s}: {mean_acc:.2f}%")


def generate_kaggle_submission(model, test_loader, device, output_path="submission.csv"):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            eeg = batch["eeg"].to(device)
            subject_idx = batch["subject_idx"].to(device)
            logits = model(eeg, subject_idx)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Category"])
        for i, pred in enumerate(all_preds):
            writer.writerow([i, IDX_TO_CAT[pred]])
    print(f"Kaggle submission saved to {output_path} ({len(all_preds)} predictions)")


def plot_confusion_matrix_text(cm):
    num_classes = cm.shape[0]
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8) * 100
    print("\nConfusion Matrix (% per true class):")
    header = "          " + " ".join([f"{IDX_TO_CAT[j][:4]:>5s}" for j in range(num_classes)])
    print(header)
    for i in range(num_classes):
        row = f"{IDX_TO_CAT[i][:8]:>8s}: "
        row += " ".join([f"{cm_norm[i, j]:5.1f}" for j in range(num_classes)])
        print(row)
