"""
evaluate_recognition_knn.py
------------------------------
Trains and evaluates a K-Nearest Neighbours (KNN) classifier on top of
each proposal-spec embedding extractor (ResNet-50, MobileNetV2, FaceNet —
from feature_extractors.py). KNeighborsClassifier.fit() doesn't solve an
optimisation problem the way SVM does — it mainly stores the training
embeddings — but the real "learning" happens at prediction time, when it
looks at the K closest stored embeddings (K=3 here) to a new face and
takes a majority vote on identity. It's still trained on your
eval_data/classifier/train data and evaluated the same way as SVM, just
via a different mechanism.

This is a SEPARATE model from SVM (see evaluate_recognition_svm.py) — SVM
and KNN are two fundamentally different algorithms, not two settings of
the same one, so they are evaluated independently here rather than in one
combined script.

IMPORTANT — this uses eval_data/classifier/train + eval_data/classifier/test,
NOT eval_data/recognition/gallery + eval_data/recognition/probe. The
gallery/probe set is reserved for the untrained nearest-neighbour
comparison in evaluate_recognition.py; reusing it here would mean the
"trained" classifier and the "untrained" comparison are scored on
overlapping data, which would muddy the comparison between the two
approaches. Capture the classifier set separately with
capture_eval_dataset.py (mode 2).

Method:
  1. For each embedding extractor (ResNet-50, MobileNetV2, FaceNet):
     a. Extract embeddings for every classifier/train image -> X_train, y_train
     b. Extract embeddings for every classifier/test image  -> X_test,  y_test
     c. TRAIN a KNN classifier (k=3) on (X_train, y_train)
     d. Predict on X_test, compare against y_test
     e. Report accuracy / precision / recall / F1 / confusion matrix,
        plus training time and per-image prediction latency

Requires: eval_data/classifier/train/<person>/*.jpg,
          eval_data/classifier/test/<person>/*.jpg
          pip install tensorflow deepface scikit-learn matplotlib --break-system-packages

Output: results/recognition_knn_<extractor>_predictions.csv
        results/confusion_matrix_knn_<extractor>.png
        results/recognition_knn_summary.csv
"""

import os
import time

from eval_utils import CLASSIFIER_TRAIN_DIR, CLASSIFIER_TEST_DIR, RESULTS_DIR, write_csv, check_data_present
from feature_extractors import EXTRACTORS
from classifier_eval_common import extract_split_embeddings, compute_metrics, plot_confusion_matrix

K_NEIGHBORS = 3


def build_knn():
    from sklearn.neighbors import KNeighborsClassifier
    return KNeighborsClassifier(n_neighbors=K_NEIGHBORS)


def main():
    print("=== KNN Recognition Evaluation ===")
    print(f"Architectures: ResNet-50 / MobileNetV2 / FaceNet  |  Classifier: KNN, k={K_NEIGHBORS} (trained)")
    print("Dataset: eval_data/classifier/train + eval_data/classifier/test\n")

    ok, message = check_data_present(CLASSIFIER_TRAIN_DIR, CLASSIFIER_TEST_DIR, min_people=2, min_images_per_person=2)
    if not ok:
        print(message)
        print("\nUse capture_eval_dataset.py (mode 2) to build the classifier train/test split first.")
        return

    summary_rows = []

    for extractor_name, extractor_cls in EXTRACTORS.items():
        print(f"\n--- Loading {extractor_name} and building embeddings ---")
        try:
            extractor = extractor_cls()
        except Exception as e:
            print(f"  [error] could not load {extractor_name}: {e}")
            print("  Skipping. Check that tensorflow / deepface are installed "
                  "(pip install tensorflow deepface --break-system-packages).")
            continue

        print("  Extracting classifier/train (training) embeddings...")
        X_train, y_train, _ = extract_split_embeddings(extractor, CLASSIFIER_TRAIN_DIR)
        print("  Extracting classifier/test (test) embeddings...")
        X_test, y_test, test_paths = extract_split_embeddings(extractor, CLASSIFIER_TEST_DIR)

        if len(X_train) == 0 or len(X_test) == 0:
            print(f"  [warn] no usable embeddings extracted for {extractor_name} — skipping.")
            continue

        print(f"  Training KNN (k={K_NEIGHBORS}) on {extractor_name} embeddings "
              f"({len(X_train)} samples, {len(set(y_train))} people)...")
        clf = build_knn()

        start = time.perf_counter()
        clf.fit(X_train, y_train)  # <-- stores training embeddings; real work happens at predict time
        train_time_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        y_pred = clf.predict(X_test)
        predict_time_ms = (time.perf_counter() - start) * 1000.0
        per_image_latency_ms = predict_time_ms / len(X_test) if len(X_test) else 0

        accuracy, precision, recall, f1, cm, labels = compute_metrics(y_test, y_pred)

        rows = [
            {
                "extractor": extractor_name,
                "true_label": yt,
                "predicted_label": yp,
                "correct": yt == yp,
                "image": os.path.basename(p),
            }
            for yt, yp, p in zip(y_test, y_pred, test_paths)
        ]
        write_csv(
            os.path.join(RESULTS_DIR, f"recognition_knn_{extractor_name}_predictions.csv"),
            rows,
            fieldnames=["extractor", "true_label", "predicted_label", "correct", "image"],
        )
        plot_confusion_matrix(
            cm, labels, f"KNN (k={K_NEIGHBORS}) — {extractor_name}",
            os.path.join(RESULTS_DIR, f"confusion_matrix_knn_{extractor_name}.png"),
            cmap="Oranges",
        )

        summary_rows.append({
            "extractor": extractor_name,
            "classifier": f"KNN(k={K_NEIGHBORS})",
            "accuracy": round(accuracy * 100, 2),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f1": round(f1 * 100, 2),
            "train_time_ms": round(train_time_ms, 2),
            "avg_predict_latency_ms": round(per_image_latency_ms, 2),
            "n_train": len(X_train),
            "n_test": len(X_test),
        })
        print(f"    -> accuracy={accuracy*100:.2f}%  precision={precision*100:.2f}%  "
              f"recall={recall*100:.2f}%  f1={f1*100:.2f}%  "
              f"(train={train_time_ms:.1f}ms, predict={per_image_latency_ms:.2f}ms/img)")

    if summary_rows:
        write_csv(
            os.path.join(RESULTS_DIR, "recognition_knn_summary.csv"),
            summary_rows,
            fieldnames=["extractor", "classifier", "accuracy", "precision", "recall", "f1",
                        "train_time_ms", "avg_predict_latency_ms", "n_train", "n_test"],
        )
        print("\n=== KNN Summary — all extractors ===")
        for r in summary_rows:
            print(f"{r['extractor']:12s} | acc={r['accuracy']}% | f1={r['f1']}% | "
                  f"train={r['train_time_ms']}ms | predict={r['avg_predict_latency_ms']}ms/img")
    else:
        print("\nNo results produced. Check that tensorflow and deepface installed correctly "
              "and that eval_data/classifier/train + eval_data/classifier/test have real images.")


if __name__ == "__main__":
    main()
