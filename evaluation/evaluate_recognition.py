"""
evaluate_recognition.py
-------------------------
Compares recognition/embedding models against your gallery/probe split
(Section 16 "Recognition Model Comparison" + Section 18 "Performance
Metrics" in the proposal).

Your proposal names ResNet-50 / MobileNetV2 / FaceNet specifically.
DeepFace doesn't ship those three exact names, but gives purpose-equivalent
CNN backbones — this script compares VGG-Face (what your production
face_engine.py uses today), Facenet512 (matches "FaceNet" from your
proposal directly, and is the model to switch to if you want a genuine
128-d-style embedding claim — note Facenet512 outputs 512-d, plain
"Facenet" outputs 128-d), and ArcFace (ResNet-based, matches the
"ResNet-50" comparison point). Say so explicitly in your report: these are
justified substitutions for the named architectures, not the same models.

Method (matches your proposal's 80/20 split):
  1. Build embeddings for every image in eval_data/gallery/<person>/
  2. For every image in eval_data/probe/<person>/, embed it and find the
     nearest gallery embedding (cosine distance)
  3. If nearest distance <= model's DeepFace threshold -> predicted = that
     person's name, else -> predicted = "Unknown"
  4. Compare predicted vs true label -> accuracy / precision / recall / F1
     / confusion matrix, plus per-image latency

Requires: eval_data/recognition/gallery/<person>/*.jpg,
          eval_data/recognition/probe/<person>/*.jpg
(use capture_eval_dataset.py mode 1 to build this)

Note: this is a SEPARATE dataset from eval_data/classifier/train+test used
by evaluate_recognition_svm.py / evaluate_recognition_knn.py — this script
does no training (nearest-neighbour against a fixed DeepFace threshold),
those scripts do (SVM / KNN fit on the training split).

Output per model: results/recognition_<model>_predictions.csv,
                   results/confusion_matrix_<model>.png
Overall:           results/recognition_summary.csv
"""

import os
import time

import numpy as np

from eval_utils import GALLERY_DIR, PROBE_DIR, RESULTS_DIR, write_csv, list_people, list_images, check_data_present

MODELS_TO_COMPARE = ["VGG-Face", "Facenet512", "ArcFace"]
# Add "Facenet" (128-d, matches your proposal's literal "128-dimensional" claim)
# or "SFace" if you want a wider comparison — just append to this list.


def cosine_distance(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def build_gallery_embeddings(model_name):
    """Returns list of (person_name, embedding_vector)."""
    from deepface import DeepFace
    embeddings = []
    for person in list_people(GALLERY_DIR):
        for img_path in list_images(os.path.join(GALLERY_DIR, person)):
            try:
                reps = DeepFace.represent(
                    img_path=img_path,
                    model_name=model_name,
                    detector_backend="skip",  # gallery images are already cropped 160x160 faces
                    enforce_detection=False,
                )
                if reps:
                    embeddings.append((person, reps[0]["embedding"]))
            except Exception as e:
                print(f"  [warn] failed to embed {img_path}: {e}")
    return embeddings


def find_deepface_threshold(model_name):
    """DeepFace publishes recommended cosine-distance thresholds per model."""
    try:
        from deepface.commons import distance as dst
        return dst.findThreshold(model_name, "cosine")
    except Exception:
        return 0.40  # reasonable generic fallback if the lookup fails


def evaluate_model(model_name):
    from deepface import DeepFace

    print(f"\n=== Evaluating recognition model: {model_name} ===")
    print("Building gallery embeddings...")
    gallery = build_gallery_embeddings(model_name)
    if not gallery:
        print(f"  No gallery embeddings built for {model_name} — skipping.")
        return None

    threshold = find_deepface_threshold(model_name)
    print(f"  Gallery size: {len(gallery)} images, "
          f"{len(set(p for p, _ in gallery))} people. Threshold={threshold:.3f}")

    rows = []
    y_true, y_pred = [], []

    for person in list_people(PROBE_DIR):
        for img_path in list_images(os.path.join(PROBE_DIR, person)):
            start = time.perf_counter()
            try:
                reps = DeepFace.represent(
                    img_path=img_path,
                    model_name=model_name,
                    detector_backend="skip",
                    enforce_detection=False,
                )
                probe_emb = reps[0]["embedding"] if reps else None
            except Exception as e:
                probe_emb = None
                print(f"  [warn] failed to embed probe {img_path}: {e}")

            predicted = "Unknown"
            best_dist = None
            if probe_emb is not None:
                distances = [(gp, cosine_distance(probe_emb, ge)) for gp, ge in gallery]
                gallery_person, best_dist = min(distances, key=lambda t: t[1])
                if best_dist <= threshold:
                    predicted = gallery_person

            elapsed_ms = (time.perf_counter() - start) * 1000.0

            y_true.append(person)
            y_pred.append(predicted)
            rows.append({
                "model": model_name,
                "true_label": person,
                "predicted_label": predicted,
                "distance": round(best_dist, 4) if best_dist is not None else "",
                "correct": predicted == person,
                "latency_ms": round(elapsed_ms, 2),
                "image": os.path.basename(img_path),
            })
            print(f"  {person}/{os.path.basename(img_path)} -> {predicted} "
                  f"(dist={best_dist:.3f} thr={threshold:.3f})" if best_dist is not None
                  else f"  {person}/{os.path.basename(img_path)} -> embedding failed")

    write_csv(
        os.path.join(RESULTS_DIR, f"recognition_{model_name}_predictions.csv"),
        rows,
        fieldnames=["model", "true_label", "predicted_label", "distance",
                    "correct", "latency_ms", "image"],
    )

    metrics = compute_metrics(y_true, y_pred, model_name)
    avg_latency = sum(r["latency_ms"] for r in rows) / len(rows) if rows else 0
    metrics["avg_latency_ms"] = round(avg_latency, 2)
    metrics["n_probe_images"] = len(rows)
    return metrics


def compute_metrics(y_true, y_pred, model_name):
    try:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    except ImportError:
        print("  [warn] scikit-learn not installed — run: pip install scikit-learn --break-system-packages")
        return {"model": model_name, "accuracy": None, "precision": None, "recall": None, "f1": None}

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plot_confusion_matrix(cm, labels, model_name)

    return {
        "model": model_name,
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2),
    }


def plot_confusion_matrix(cm, labels, model_name):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [warn] matplotlib not installed — skipping confusion matrix plot. "
              "Run: pip install matplotlib --break-system-packages")
        return

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.6), max(5, len(labels) * 0.6)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name}")

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f"confusion_matrix_{model_name}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [saved] {out_path}")


def main():
    print("=== Recognition Model Comparison ===")
    ok, message = check_data_present(GALLERY_DIR, PROBE_DIR, min_people=2, min_images_per_person=2)
    if not ok:
        print(f"\n{message}")
        print("\nUse capture_eval_dataset.py (mode 1) to build the gallery/probe split first.")
        return

    all_metrics = []
    for model_name in MODELS_TO_COMPARE:
        metrics = evaluate_model(model_name)
        if metrics:
            all_metrics.append(metrics)

    if all_metrics:
        write_csv(
            os.path.join(RESULTS_DIR, "recognition_summary.csv"),
            all_metrics,
            fieldnames=["model", "accuracy", "precision", "recall", "f1",
                        "avg_latency_ms", "n_probe_images"],
        )
        print("\n=== Summary ===")
        for m in all_metrics:
            print(f"{m['model']:12s} | acc={m['accuracy']}% | prec={m['precision']}% | "
                  f"rec={m['recall']}% | f1={m['f1']}% | {m['avg_latency_ms']}ms/img")


if __name__ == "__main__":
    main()
