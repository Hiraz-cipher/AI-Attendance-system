"""
generate_report.py
---------------------
Run this LAST, after evaluate_detectors.py, evaluate_recognition.py, and
evaluate_liveness.py have all produced their summary CSVs. Pulls all three
into one Markdown report with a final recommended configuration — this is
the "conclusion" document for your dissertation/report chapter.

Output: results/REPORT.md
"""

import os
import csv

from eval_utils import RESULTS_DIR


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def best_by(rows, key, as_float=True, highest_is_best=True):
    valid = [r for r in rows if r.get(key) not in (None, "")]
    if not valid:
        return None
    conv = float if as_float else str
    return max(valid, key=lambda r: conv(r[key])) if highest_is_best else min(valid, key=lambda r: conv(r[key]))


def main():
    detector_summary = read_csv(os.path.join(RESULTS_DIR, "detector_summary.csv"))
    recognition_summary = read_csv(os.path.join(RESULTS_DIR, "recognition_summary.csv"))
    svm_summary = read_csv(os.path.join(RESULTS_DIR, "recognition_svm_summary.csv"))
    knn_summary = read_csv(os.path.join(RESULTS_DIR, "recognition_knn_summary.csv"))
    liveness_summary = read_csv(os.path.join(RESULTS_DIR, "liveness_summary.csv"))

    lines = []
    lines.append("# Model & Component Comparison Report\n")
    lines.append("Auto-generated from evaluation/results/*.csv. Review and edit before including in your final report — ")
    lines.append("state your actual dataset size and collection conditions here, since the conclusions below are only as strong as that data.\n")

    # --- Detector section ---
    lines.append("## 1. Face Detector Comparison\n")
    if detector_summary:
        lines.append("| Backend | Test type | Condition | N images | Detection rate | Avg FPS |")
        lines.append("|---|---|---|---|---|---|")
        for r in detector_summary:
            lines.append(f"| {r['backend']} | {r['test_type']} | {r['condition']} | {r['n_images']} | "
                          f"{r['detection_rate_pct']}% | {r['avg_fps']} |")
        best = best_by(detector_summary, "detection_rate_pct")
        fastest = best_by(detector_summary, "avg_fps")
        lines.append(f"\n**Highest detection rate:** {best['backend']} ({best['detection_rate_pct']}% on {best['condition']})")
        lines.append(f"\n**Fastest:** {fastest['backend']} ({fastest['avg_fps']} FPS on {fastest['condition']})")
    else:
        lines.append("_No detector comparison data yet — run evaluate_detectors.py first._")
    lines.append("")

    # --- Recognition section ---
    lines.append("## 2. Recognition Model Comparison\n")
    if recognition_summary:
        lines.append("| Model | Accuracy | Precision | Recall | F1 | Avg latency (ms) | N probe images |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in recognition_summary:
            lines.append(f"| {r['model']} | {r['accuracy']}% | {r['precision']}% | {r['recall']}% | "
                          f"{r['f1']}% | {r['avg_latency_ms']} | {r['n_probe_images']} |")
        best_acc = best_by(recognition_summary, "accuracy")
        best_latency = best_by(recognition_summary, "avg_latency_ms", highest_is_best=False)
        lines.append(f"\n**Highest accuracy:** {best_acc['model']} ({best_acc['accuracy']}%)")
        lines.append(f"\n**Lowest latency:** {best_latency['model']} ({best_latency['avg_latency_ms']}ms/image)")
        proposal_target_met = [r for r in recognition_summary if r["accuracy"] and float(r["accuracy"]) >= 98]
        if proposal_target_met:
            lines.append(f"\nModels meeting the proposal's >98% accuracy target: "
                          f"{', '.join(r['model'] for r in proposal_target_met)}")
        else:
            lines.append("\nNo model met the proposal's >98% accuracy target on this dataset — "
                          "worth discussing as a finding (likely dataset-size limited; note this honestly).")
    else:
        lines.append("_No recognition comparison data yet — run evaluate_recognition.py first._")
    lines.append("")

    # --- Proposal-spec model comparison: SVM (real architectures + trained classifier) ---
    lines.append("## 2b. Proposal-Spec Recognition — SVM (ResNet-50 / MobileNetV2 / FaceNet, trained)\n")
    if svm_summary:
        lines.append("| Extractor | Accuracy | Precision | Recall | F1 | Train time (ms) | Predict latency (ms/img) |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in svm_summary:
            lines.append(f"| {r['extractor']} | {r['accuracy']}% | {r['precision']}% | "
                          f"{r['recall']}% | {r['f1']}% | {r['train_time_ms']} | {r['avg_predict_latency_ms']} |")
        best_svm = best_by(svm_summary, "accuracy")
        lines.append(f"\n**Best SVM combination:** {best_svm['extractor']} ({best_svm['accuracy']}% accuracy)")
    else:
        lines.append("_No SVM data yet — run evaluate_recognition_svm.py first "
                      "(requires tensorflow + deepface + scikit-learn installed)._")
    lines.append("")

    # --- Proposal-spec model comparison: KNN (real architectures + trained classifier) ---
    lines.append("## 2c. Proposal-Spec Recognition — KNN (ResNet-50 / MobileNetV2 / FaceNet, trained)\n")
    if knn_summary:
        lines.append("| Extractor | Accuracy | Precision | Recall | F1 | Train time (ms) | Predict latency (ms/img) |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in knn_summary:
            lines.append(f"| {r['extractor']} | {r['accuracy']}% | {r['precision']}% | "
                          f"{r['recall']}% | {r['f1']}% | {r['train_time_ms']} | {r['avg_predict_latency_ms']} |")
        best_knn = best_by(knn_summary, "accuracy")
        lines.append(f"\n**Best KNN combination:** {best_knn['extractor']} ({best_knn['accuracy']}% accuracy)")
    else:
        lines.append("_No KNN data yet — run evaluate_recognition_knn.py first "
                      "(requires tensorflow + deepface + scikit-learn installed)._")
    lines.append("")

    if svm_summary and knn_summary:
        overall_best_svm = best_by(svm_summary, "accuracy")
        overall_best_knn = best_by(knn_summary, "accuracy")
        winner = "SVM" if float(overall_best_svm["accuracy"]) >= float(overall_best_knn["accuracy"]) else "KNN"
        lines.append(
            f"**SVM vs KNN:** best SVM result was {overall_best_svm['accuracy']}% "
            f"({overall_best_svm['extractor']}); best KNN result was {overall_best_knn['accuracy']}% "
            f"({overall_best_knn['extractor']}). **{winner}** performed better overall on this dataset — "
            "note this is likely sensitive to dataset size, since KNN's neighbour-voting approach "
            "tends to need more images per person than SVM's margin-based approach to perform well."
        )
        lines.append("")

    lines.append(
        "Note: unlike Section 2 above (nearest-neighbour distance against DeepFace's built-in threshold, "
        "no training involved, evaluated on eval_data/recognition/gallery+probe), both of these sections "
        "come from an actually **trained** classifier fitted on eval_data/classifier/train embeddings and "
        "scored on eval_data/classifier/test — a separate capture session from the gallery/probe set. "
        "This is the literal architecture your proposal describes (CNN embedding + trained classifier)."
    )
    lines.append("")

    # --- Liveness section ---
    lines.append("## 3. Liveness / Anti-Spoofing Evaluation\n")
    if liveness_summary:
        lines.append("| Metric | Value | N samples |")
        lines.append("|---|---|---|")
        for r in liveness_summary:
            lines.append(f"| {r['metric']} | {r['value_pct']}% | {r['n_samples']} |")
        far_row = next((r for r in liveness_summary if r["metric"].startswith("FAR overall")), None)
        if far_row and far_row["value_pct"]:
            far_val = float(far_row["value_pct"])
            verdict = "meets" if far_val < 2 else "does not meet"
            lines.append(f"\nOverall measured FAR is {far_val}%, which **{verdict}** the proposal's <2% target.")
    else:
        lines.append("_No liveness evaluation data yet — run evaluate_liveness.py first._")
    lines.append("")

    # --- Final recommendation ---
    lines.append("## 4. Recommended Configuration\n")
    if detector_summary and recognition_summary and liveness_summary:
        best_detector = best_by(detector_summary, "detection_rate_pct")
        best_model = best_by(recognition_summary, "accuracy")
        lines.append(
            f"Based on the results above, the recommended production configuration is "
            f"**{best_detector['backend']} for face detection** combined with "
            f"**{best_model['model']} for recognition**, trading off "
            f"accuracy ({best_model['accuracy']}%) against latency ({best_model['avg_latency_ms']}ms/image) "
            f"for real-time suitability at your target frame rate.\n"
        )
        lines.append(
            "_Edit this paragraph once you have real numbers — the script picks whichever model has "
            "the single highest accuracy figure, but you should weigh latency, FPS, and how the "
            "confusion matrices look for your actual write-up rather than trusting this line blindly._"
        )
    else:
        lines.append("_Run all three evaluation scripts first, then re-run this script to get a full recommendation._")

    report_path = os.path.join(RESULTS_DIR, "REPORT.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[saved] {report_path}")


if __name__ == "__main__":
    main()
