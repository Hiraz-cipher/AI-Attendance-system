"""
feature_extractors.py
------------------------
Builds embeddings using the EXACT architectures your proposal names —
ResNet-50, MobileNetV2, and FaceNet — instead of evaluate_recognition.py's
DeepFace substitutes (VGG-Face/Facenet512/ArcFace). This is what lets you
genuinely say "I compared ResNet-50, MobileNetV2, and FaceNet," not "I
compared DeepFace's available backbones."

IMPORTANT HONESTY NOTE FOR YOUR REPORT:
ResNet-50 and MobileNetV2 here are loaded with their standard, publicly
available ImageNet-pretrained weights, with the classification head
removed and global-average-pooling applied to produce a fixed-length
vector. They were NOT trained specifically for face recognition — unlike
FaceNet, which was trained specifically on a face-verification task
(triplet loss over paired face images). Using general ImageNet CNN
backbones as feature extractors is a legitimate, literature-supported
technique (transfer learning / off-the-shelf feature extraction), but it
means ResNet-50 and MobileNetV2 embeddings encode general visual features,
not face-identity-specific features the way FaceNet's do. If your results
show FaceNet clearly outperforming the other two, THAT GAP ITSELF is a
valid, reportable finding — not something to hide.

FaceNet here is the real thing: DeepFace's "Facenet" model (not
"Facenet512") is Google's actual FaceNet architecture with its published
128-dimensional embedding — a genuine match to your proposal's model name
and its "128-dimensional embedding" claim, not a substitute.

Requires:
    pip install tensorflow deepface --break-system-packages
"""

import numpy as np
import cv2


def _preprocess_for_keras(image_bgr, target_size, preprocess_fn):
    img = cv2.resize(image_bgr, target_size)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32")
    img_rgb = np.expand_dims(img_rgb, axis=0)
    return preprocess_fn(img_rgb)


class ResNet50Extractor:
    """ImageNet-pretrained ResNet-50, used as a generic feature extractor
    (classification head removed, global-average-pooled to a 1D vector)."""
    name = "ResNet-50"

    def __init__(self):
        from tensorflow.keras.applications import ResNet50
        from tensorflow.keras.applications.resnet50 import preprocess_input
        from tensorflow.keras.layers import GlobalAveragePooling2D
        from tensorflow.keras.models import Model

        base = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        pooled = GlobalAveragePooling2D()(base.output)
        self.model = Model(inputs=base.input, outputs=pooled)
        self.preprocess_fn = preprocess_input
        self.target_size = (224, 224)

    def embed(self, image_bgr):
        x = _preprocess_for_keras(image_bgr, self.target_size, self.preprocess_fn)
        return self.model.predict(x, verbose=0)[0]


class MobileNetV2Extractor:
    """ImageNet-pretrained MobileNetV2, used as a generic feature extractor
    — the lightweight architecture named in the proposal for CPU-friendly
    deployment."""
    name = "MobileNetV2"

    def __init__(self):
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        from tensorflow.keras.layers import GlobalAveragePooling2D
        from tensorflow.keras.models import Model

        base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        pooled = GlobalAveragePooling2D()(base.output)
        self.model = Model(inputs=base.input, outputs=pooled)
        self.preprocess_fn = preprocess_input
        self.target_size = (224, 224)

    def embed(self, image_bgr):
        x = _preprocess_for_keras(image_bgr, self.target_size, self.preprocess_fn)
        return self.model.predict(x, verbose=0)[0]


class FaceNetExtractor:
    """The real FaceNet architecture (128-d embedding), via DeepFace's
    "Facenet" model — genuinely matches the proposal's named model and its
    128-dimensional embedding claim, unlike Facenet512 (512-d) used
    elsewhere in this evaluation suite as a substitute."""
    name = "FaceNet"

    def __init__(self):
        from deepface import DeepFace
        DeepFace.build_model("Facenet")
        self._DeepFace = DeepFace

    def embed(self, image_bgr):
        reps = self._DeepFace.represent(
            img_path=image_bgr,
            model_name="Facenet",
            detector_backend="skip",
            enforce_detection=False,
        )
        return np.asarray(reps[0]["embedding"]) if reps else None


# Order matters only for display/logging — dict preserves insertion order in Python 3.7+
EXTRACTORS = {
    "ResNet-50": ResNet50Extractor,
    "MobileNetV2": MobileNetV2Extractor,
    "FaceNet": FaceNetExtractor,
}
