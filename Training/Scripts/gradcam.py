"""
==============================================================
TruthLens Grad-CAM Visualization
Version : 1.0
==============================================================
"""

import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from tensorflow import keras
import matplotlib.pyplot as plt

# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_DIR = Path(r"D:\TruthLens")

MODEL_PATH = PROJECT_DIR / "Models" / "truthlens_efficientnetb0_finetuned.keras"

TEST_DIR = PROJECT_DIR / "Final_Dataset" / "Test"

RESULT_DIR = PROJECT_DIR / "Results" / "GradCAM"

ORIGINAL_DIR = RESULT_DIR / "Original"

HEATMAP_DIR = RESULT_DIR / "Heatmaps"

OVERLAY_DIR = RESULT_DIR / "Overlay"

REPORT_FILE = RESULT_DIR / "prediction_report.txt"

# ==========================================================
# CREATE OUTPUT FOLDERS
# ==========================================================

ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# SETTINGS
# ==========================================================

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "AI_Generated",
    "Deepfake",
    "Manipulated",
    "Real",
]

LAST_CONV_LAYER = "top_conv"

# ==========================================================
# LOAD MODEL
# ==========================================================

print("=" * 70)
print("TRUTHLENS GRAD-CAM")
print("=" * 70)

print("\nLoading Fine Tuned Model...\n")

model = keras.models.load_model(MODEL_PATH)

print("Model Loaded Successfully.")

print("\nModel Summary\n")

model.summary()

# ==========================================================
# FIND EFFICIENTNET
# ==========================================================

base_model = None

for layer in model.layers:

    if "efficientnet" in layer.name.lower():

        base_model = layer

        break

if base_model is None:

    raise Exception("EfficientNet not found.")

print("\nEfficientNet Found")

print("Last Conv Layer :", LAST_CONV_LAYER)
# ==========================================================
# IMAGE PREPROCESSING
# ==========================================================

def preprocess_image(image_path):

    image = cv2.imread(str(image_path))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    original = image.copy()

    image = cv2.resize(image, IMAGE_SIZE)

    image = image.astype("float32")

    image = keras.applications.efficientnet.preprocess_input(image)

    image = np.expand_dims(image, axis=0)

    return image, original


# ==========================================================
# SAVE IMAGE
# ==========================================================

def save_rgb(path, image):

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(path), image)


# ==========================================================
# SOFTMAX PREDICTION
# ==========================================================

def predict(image):

    prediction = model.predict(image, verbose=0)

    index = np.argmax(prediction[0])

    confidence = float(prediction[0][index])

    return index, confidence, prediction
# ==========================================================
# BUILD GRAD-CAM MODEL
# ==========================================================

print("\nBuilding Grad-CAM Model...")

last_conv_layer = base_model.get_layer(LAST_CONV_LAYER)

# Create a model that outputs the feature maps
conv_model = keras.Model(
    inputs=base_model.input,
    outputs=last_conv_layer.output
)

print("Grad-CAM Model Ready.")


# ==========================================================
# GENERATE HEATMAP
# ==========================================================

def make_gradcam_heatmap(image, pred_index=None):

    # Forward pass through EfficientNet
    with tf.GradientTape() as tape:

        feature_maps = conv_model(image)

        tape.watch(feature_maps)

        x = feature_maps

        # Remaining layers after EfficientNet
        x = model.layers[3](x)   # GlobalAveragePooling2D
        x = model.layers[4](x)   # Dropout
        predictions = model.layers[5](x)  # Dense

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, feature_maps)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    feature_maps = feature_maps[0]

    heatmap = feature_maps @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)

    heatmap /= tf.reduce_max(heatmap) + 1e-10

    return heatmap.numpy()

# ==========================================================
# CREATE COLOR HEATMAP
# ==========================================================

def create_heatmap_image(heatmap, original_image):

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.resize(
        heatmap,
        (
            original_image.shape[1],
            original_image.shape[0]
        )
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    return heatmap


# ==========================================================
# CREATE OVERLAY
# ==========================================================

def overlay_heatmap(original, heatmap, alpha=0.45):

    overlay = cv2.addWeighted(
        original,
        1 - alpha,
        heatmap,
        alpha,
        0
    )

    return overlay
# ==========================================================
# PROCESS SINGLE IMAGE
# ==========================================================

def process_image(image_path):

    print("\n--------------------------------------------------")
    print("Processing:", image_path.name)

    image, original = preprocess_image(image_path)

    class_id, confidence, raw_prediction = predict(image)

    predicted_class = CLASS_NAMES[class_id]

    print("Prediction :", predicted_class)
    print(f"Confidence : {confidence*100:.2f}%")

    # ---------------------------------------------
    # Generate Grad-CAM
    # ---------------------------------------------

    heatmap = make_gradcam_heatmap(
        image,
        pred_index=class_id
    )

    heatmap_image = create_heatmap_image(
        heatmap,
        original
    )

    overlay = overlay_heatmap(
        original,
        heatmap_image,
        alpha=0.45
    )

    # ---------------------------------------------
    # Save Images
    # ---------------------------------------------

    original_path = ORIGINAL_DIR / image_path.name

    heatmap_path = HEATMAP_DIR / image_path.name

    overlay_path = OVERLAY_DIR / image_path.name

    save_rgb(
        original_path,
        original
    )

    save_rgb(
        heatmap_path,
        heatmap_image
    )

    save_rgb(
        overlay_path,
        overlay
    )

    # ---------------------------------------------
    # Report
    # ---------------------------------------------

    with open(REPORT_FILE, "a") as report:

        report.write("=" * 70 + "\n")

        report.write(f"Image      : {image_path.name}\n")

        report.write(f"Prediction : {predicted_class}\n")

        report.write(f"Confidence : {confidence*100:.2f}%\n\n")

        report.write("Class Probabilities\n")

        for i, cls in enumerate(CLASS_NAMES):

            report.write(
                f"{cls:15} : {raw_prediction[0][i]*100:.2f}%\n"
            )

        report.write("\n")

        report.write(f"Original : {original_path}\n")

        report.write(f"Heatmap : {heatmap_path}\n")

        report.write(f"Overlay : {overlay_path}\n")

        report.write("\n\n")

    print("Saved Original :", original_path.name)
    print("Saved Heatmap  :", heatmap_path.name)
    print("Saved Overlay  :", overlay_path.name)
    # ==========================================================
# RESET REPORT
# ==========================================================

with open(REPORT_FILE, "w") as report:

    report.write("TruthLens Grad-CAM Report\n")

    report.write("=" * 70 + "\n\n")

print("\nPrediction report initialized.")
# ==========================================================
# GET RANDOM IMAGE FROM EACH CLASS
# ==========================================================

import random

random.seed(42)

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


def get_random_image(class_name):

    class_folder = TEST_DIR / class_name

    if not class_folder.exists():

        print(f"Folder not found: {class_folder}")

        return None

    images = [
        file
        for file in class_folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in VALID_EXTENSIONS
    ]

    if not images:

        print(f"No images found in {class_name}")

        return None

    return random.choice(images)


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("\n" + "=" * 70)
    print("TRUTHLENS GRAD-CAM VISUALIZATION")
    print("=" * 70)

    processed = 0

    for class_name in CLASS_NAMES:

        print(f"\nSelecting image from {class_name}...")

        image_path = get_random_image(class_name)

        if image_path is None:
            continue

        process_image(image_path)

        processed += 1

    print("\n" + "=" * 70)
    print("GRAD-CAM COMPLETED")
    print("=" * 70)

    print(f"\nImages Processed : {processed}")

    print("\nResults saved to:")

    print(RESULT_DIR)

    print("\nGenerated Files")

    print("----------------------------")

    print("Original Images")

    print("Heatmaps")

    print("Overlay Images")

    print("Prediction Report")

    print("\nTruthLens Explainable AI completed successfully.")


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    main()