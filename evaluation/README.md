# Evaluation Harness

Standalone comparison suite for the AI/ML evaluation section of the project
proposal (Section 16 & 18). Lives entirely in this `evaluation/` folder —
nothing here modifies `app.py`, `face_engine.py`, `liveness.py`, the
production `dataset/` folder, or `attendance.db`. It only *imports* from
`liveness.py` (unmodified) so the liveness numbers reflect exactly what the
real app runs.

## Folder layout

```
evaluation/
├── eval_data/
│   ├── gallery/<person>/*.jpg        80% split — recognition "registered" images
│   ├── probe/<person>/*.jpg          20% split — recognition held-out test images
│   ├── spoof/printed_photo/<person>/ printed-photo spoof attempts
│   ├── spoof/screen_replay/<person>/ phone/tablet screen-replay spoof attempts
│   ├── live_test/<person>/           genuine live footage (clip.mp4 or frames)
│   ├── lighting_test/<condition>/<person>/  normal / dim / bright, one face per image
│   └── pose_test/<condition>/<person>/      frontal / angled / profile, one face per image
├── results/                          all CSVs, confusion matrix PNGs, and REPORT.md land here
├── eval_utils.py                     shared paths + helpers
├── feature_extractors.py             REAL ResNet-50 / MobileNetV2 / FaceNet embedding extractors
├── classifier_eval_common.py         shared embedding-extraction + metrics code for the SVM/KNN scripts
├── capture_eval_dataset.py           webcam capture tool for all the folders above
├── evaluate_detectors.py             Haar vs YOLOv8 comparison
├── evaluate_recognition.py           DeepFace substitutes (VGG-Face/Facenet512/ArcFace), nearest-neighbour, NO training
├── evaluate_recognition_svm.py       proposal-exact models (ResNet-50/MobileNetV2/FaceNet) + TRAINED SVM classifier
├── evaluate_recognition_knn.py       proposal-exact models (ResNet-50/MobileNetV2/FaceNet) + TRAINED KNN classifier
├── evaluate_liveness.py              TAR / FAR / FRR against your real LivenessDetector
└── generate_report.py                combines all results into results/REPORT.md
```

## Recognition evaluation paths — three separate scripts, three separate questions

| | `evaluate_recognition.py` | `evaluate_recognition_svm.py` | `evaluate_recognition_knn.py` |
|---|---|---|---|
| Models | VGG-Face, Facenet512, ArcFace (DeepFace's available backbones) | ResNet-50, MobileNetV2, FaceNet — **the exact names your proposal states** | Same three, exact names |
| Matching/Classifier | Nearest-neighbour cosine distance against DeepFace's built-in threshold | Support Vector Machine — finds a max-margin decision boundary | K-Nearest Neighbours (k=3) — majority vote among closest stored embeddings |
| Involves training? | No — no `.fit()` call anywhere, pure distance lookup | Yes — `SVC.fit()` solves a real optimisation problem | Yes — `.fit()` stores training embeddings; the real "learning" happens at prediction time via neighbour voting |
| Dependencies | `deepface` | `tensorflow`, `deepface`, `scikit-learn` | `tensorflow`, `deepface`, `scikit-learn` |

SVM and KNN are two fundamentally different algorithms — not two settings of the same classifier — so they're deliberately kept in **separate scripts** rather than one combined file, both importing shared plumbing from `classifier_eval_common.py`.

**Run all three if you can.** `evaluate_recognition.py` tells you how your production app's actual approach performs; `evaluate_recognition_svm.py` and `evaluate_recognition_knn.py` tell you how your proposal's literal design (CNN embedding + trained classifier) performs, and let you directly compare SVM against KNN as a finding in its own right.

Run them with:
```
python evaluate_recognition_svm.py
python evaluate_recognition_knn.py
```

**Honesty note for your report:** ResNet-50 and MobileNetV2 here use their standard ImageNet-pretrained weights as generic feature extractors — they were not trained specifically for face recognition, unlike FaceNet. If FaceNet clearly outperforms the other two in your results, that gap is itself a valid finding to report, not something to smooth over. Likewise, if SVM and KNN diverge noticeably, that's worth discussing — KNN's neighbour-voting approach is typically more sensitive to how many images per person you collected than SVM's margin-based approach.

## Workflow

1. **Collect data** with `python capture_eval_dataset.py` — run it once per
   volunteer, per mode. Start small: even 5 volunteers x 15 images is enough
   to run every comparison meaningfully. You don't need this finished before
   writing the scripts — that's why this exists now.

   - Mode 1 (gallery+probe) → needed for `evaluate_recognition.py`
   - Mode 2 (spoof) → needed for `evaluate_liveness.py` (also drop a short
     `.mp4` or frame folder per person into `eval_data/live_test/` for the
     genuine-live side of that comparison)
   - Modes 3/4 (lighting/pose) → needed for `evaluate_detectors.py`

2. **Install extra dependencies** (on top of what your app already needs):
   ```
   pip install scikit-learn matplotlib --break-system-packages
   ```
   For `evaluate_recognition_svm.py` and `evaluate_recognition_knn.py` specifically, also install:
   ```
   pip install tensorflow --break-system-packages
   ```

3. **Run the comparisons** in any order — each one checks its own data and
   tells you plainly what's missing if you're not ready yet:
   ```
   python evaluate_detectors.py
   python evaluate_recognition.py
   python evaluate_liveness.py
   ```

4. **Generate the final report**:
   ```
   python generate_report.py
   ```
   Produces `results/REPORT.md` with tables for all three comparisons plus
   a recommended final configuration — this is your Section 16/18
   "conclusion" writeup starting point. Review and edit it (especially the
   dataset-size caveats) before it goes in your actual report; don't submit
   it verbatim.

## Notes for your report

- Be upfront about dataset size and collection conditions in whatever you
  submit — a small, honestly-described dataset is legitimate; an
  overstated one isn't.
- `evaluate_recognition.py` compares `VGG-Face`, `Facenet512`, and `ArcFace`
  as justified substitutes for your proposal's ResNet-50/MobileNetV2/FaceNet
  — say so explicitly rather than implying they're the exact same models.
- If you want your production system to genuinely match the proposal's
  "128-dimensional embedding" claim, switch `MODEL_NAME` in
  `face_engine.py` to `"Facenet"` (128-d) instead of `"VGG-Face"` — that's
  a one-line change in the parent project, separate from this evaluation
  folder.
- `evaluate_liveness.py` FAR numbers only cover printed-photo and
  screen-replay attacks (matching your proposal's stated spoof test cases).
  It will not, and isn't meant to, catch a video of the person actually
  blinking played back — note that as a known limitation, not a bug.
