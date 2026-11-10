"""
classifier_eval_common.py
----------------------------
Shared helper functions used by BOTH evaluate_recognition_svm.py and
evaluate_recognition_knn.py, so the embedding-extraction and metrics code
isn't duplicated across the two files. SVM and KNN are trained and
evaluated completely independently in their own scripts — this file only
holds the identical plumbing both of them need (loading images, extracting
embeddings, computing accuracy/precision/recall/F1, plotting confusion
matrices).
"""

import os

import numpy as np
import cv2

from eval_utils import list_people, list_images


def extract_split_embeddings(extractor, root_dir):
    """Returns (X, y, image_paths) — embeddings, labels, and source paths."""
    X, y, paths = [], [], []
    for person in list_people(root_dir):
        for img_path in list_images(os.path.join(root_dir, person)):
            img = cv2.imread(img_path)
            if img is None:
                print(f"  [warn] could not read {img_path}")
                continue
            try:
                emb = extractor.embed(img)
            except Exception as e:
                print(f"  [warn] failed to embed {img_path}: {e}")
                continue
            if emb is None:
                continue
            X.append(emb)
            y.append(person)
            paths.append(img_path)
    return np.array(X), np.array(y), paths


def compute_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return accuracy, precision, recall, f1, cm, labels


def plot_confusion_matrix(cm, labels, title, out_path, cmap="Blues"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [warn] matplotlib not installed — skipping confusion matrix plot.")
        return

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.6), max(5, len(labels) * 0.6)))
    im = ax.imshow(cm, cmap=cmap)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [saved] {out_path}")
