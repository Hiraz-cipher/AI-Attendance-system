import os
from face_engine import FaceEngine

DATASET_DIR = "dataset"
MODEL_NAME = "VGG-Face"


def main():
    print("Building face embeddings from the dataset folder...")

    if not os.path.exists(DATASET_DIR) or not os.listdir(DATASET_DIR):
        print("Dataset directory is empty. Please capture some faces first.")
        return

    engine = FaceEngine(dataset_dir=DATASET_DIR, model_name=MODEL_NAME)
    print("Triggering embedding generation...")
    if engine.rebuild_embeddings():
        print("\nEmbeddings generated and saved successfully in the dataset folder!")
    else:
        print("No images found in the dataset directory.")


if __name__ == "__main__":
    main()
