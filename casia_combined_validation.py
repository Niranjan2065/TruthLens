import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# TRUTHLENS - CASIA 2.0 COMBINED EXTERNAL VALIDATION
# ============================================================

PROJECT_DIR = Path(r"D:\TruthLens")

MODEL_PATH = PROJECT_DIR / "Models" / "truthlens_model.tflite"

REAL_DIR = PROJECT_DIR / "RealWorld_CASIA_Test" / "Real"

MANIPULATED_DIR = (
    PROJECT_DIR / "RealWorld_CASIA_Test" / "Manipulated"
)

RESULTS_DIR = PROJECT_DIR / "Results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULT_FILE = (
    RESULTS_DIR / "casia_combined_validation_results.txt"
)

# ============================================================
# CLASS ORDER
# ============================================================

CLASS_NAMES = [
    "AI_Generated",
    "Deepfake",
    "Manipulated",
    "Real"
]

CLASS_TO_INDEX = {
    "AI_Generated": 0,
    "Deepfake": 1,
    "Manipulated": 2,
    "Real": 3
}

IMAGE_SIZE = (224, 224)

# ============================================================
# PRINT HEADER
# ============================================================

print("=" * 70)
print("TRUTHLENS - CASIA 2.0 COMBINED EXTERNAL VALIDATION")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)

print("\nReal dataset:")
print(REAL_DIR)

print("\nManipulated dataset:")
print(MANIPULATED_DIR)

# ============================================================
# FIND UNIQUE IMAGES
# ============================================================

def get_unique_images(folder):

    if not folder.exists():
        raise FileNotFoundError(
            f"Folder not found:\n{folder}"
        )

    images = []

    for path in folder.iterdir():

        if (
            path.is_file()
            and path.suffix.lower()
            in [".jpg", ".jpeg", ".png"]
        ):
            images.append(path)

    return sorted(set(images))


real_images = get_unique_images(REAL_DIR)

manipulated_images = get_unique_images(
    MANIPULATED_DIR
)

# Use maximum 100 from each class

real_images = real_images[:100]

manipulated_images = manipulated_images[:100]

print("\n" + "=" * 70)
print("CASIA DATASET")
print("=" * 70)

print(
    f"Real images        : {len(real_images)}"
)

print(
    f"Manipulated images : {len(manipulated_images)}"
)

total_images = (
    len(real_images)
    + len(manipulated_images)
)

print(
    f"Total images       : {total_images}"
)

# ============================================================
# LOAD TFLITE MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING TFLITE MODEL")
print("=" * 70)

interpreter = tf.lite.Interpreter(
    model_path=str(MODEL_PATH)
)

interpreter.allocate_tensors()

input_details = (
    interpreter.get_input_details()
)

output_details = (
    interpreter.get_output_details()
)

print(
    "\nInput shape:",
    input_details[0]["shape"]
)

print(
    "Input dtype:",
    input_details[0]["dtype"]
)

print(
    "Output shape:",
    output_details[0]["shape"]
)

print(
    "Output dtype:",
    output_details[0]["dtype"]
)

# ============================================================
# PREPROCESS
# ============================================================

def preprocess(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        IMAGE_SIZE
    )

    image = np.array(
        image
    ).astype(np.float32)

    # IMPORTANT:
    # Do NOT divide by 255.
    # This matches the TruthLens EfficientNetB0
    # training configuration.

    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# ============================================================
# PREDICTION
# ============================================================

def predict(image):

    interpreter.set_tensor(
        input_details[0]["index"],
        image
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_details[0]["index"]
    )

    probabilities = output[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_index]
    )

    return predicted_index, confidence


# ============================================================
# RUN EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("RUNNING CASIA COMBINED VALIDATION")
print("=" * 70)

true_labels = []

predicted_labels = []

confidences = []

# ============================================================
# REAL IMAGES
# ============================================================

print("\n--- REAL IMAGES ---")

for i, image_path in enumerate(
    real_images,
    start=1
):

    image = preprocess(
        image_path
    )

    predicted_index, confidence = predict(
        image
    )

    true_index = CLASS_TO_INDEX["Real"]

    true_labels.append(
        true_index
    )

    predicted_labels.append(
        predicted_index
    )

    confidences.append(
        confidence
    )

    print(
        f"{i:03d}/{len(real_images):03d} "
        f"{image_path.name:35} -> "
        f"{CLASS_NAMES[predicted_index]:15} "
        f"{confidence * 100:6.2f}%"
    )


# ============================================================
# MANIPULATED IMAGES
# ============================================================

print("\n--- MANIPULATED IMAGES ---")

for i, image_path in enumerate(
    manipulated_images,
    start=1
):

    image = preprocess(
        image_path
    )

    predicted_index, confidence = predict(
        image
    )

    true_index = CLASS_TO_INDEX[
        "Manipulated"
    ]

    true_labels.append(
        true_index
    )

    predicted_labels.append(
        predicted_index
    )

    confidences.append(
        confidence
    )

    print(
        f"{i:03d}/{len(manipulated_images):03d} "
        f"{image_path.name:35} -> "
        f"{CLASS_NAMES[predicted_index]:15} "
        f"{confidence * 100:6.2f}%"
    )


# ============================================================
# NUMPY ARRAYS
# ============================================================

true_labels = np.array(
    true_labels
)

predicted_labels = np.array(
    predicted_labels
)

confidences = np.array(
    confidences
)

# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

precision = precision_score(
    true_labels,
    predicted_labels,
    labels=[0, 1, 2, 3],
    average="macro",
    zero_division=0
)

recall = recall_score(
    true_labels,
    predicted_labels,
    labels=[0, 1, 2, 3],
    average="macro",
    zero_division=0
)

f1 = f1_score(
    true_labels,
    predicted_labels,
    labels=[0, 1, 2, 3],
    average="macro",
    zero_division=0
)

average_confidence = np.mean(
    confidences
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=[0, 1, 2, 3]
)

# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("CASIA COMBINED VALIDATION RESULTS")
print("=" * 70)

print(
    f"\nTotal images tested : {len(true_labels)}"
)

print(
    f"Accuracy            : {accuracy * 100:.2f}%"
)

print(
    f"Macro Precision     : {precision * 100:.2f}%"
)

print(
    f"Macro Recall        : {recall * 100:.2f}%"
)

print(
    f"Macro F1 Score      : {f1 * 100:.2f}%"
)

print(
    f"Average confidence  : "
    f"{average_confidence * 100:.2f}%"
)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    true_labels,
    predicted_labels,
    labels=[0, 1, 2, 3],
    target_names=CLASS_NAMES,
    zero_division=0
)

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report)

# ============================================================
# CONFUSION MATRIX
# ============================================================

print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(
    "\nRows    = Actual"
)

print(
    "Columns = Predicted\n"
)

print(
    f"{'':18}"
    f"{'AI_Generated':14}"
    f"{'Deepfake':12}"
    f"{'Manipulated':14}"
    f"{'Real':10}"
)

for i, row in enumerate(cm):

    print(
        f"{CLASS_NAMES[i]:18}"
        f"{row[0]:14}"
        f"{row[1]:12}"
        f"{row[2]:14}"
        f"{row[3]:10}"
    )

# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION DISTRIBUTION")
print("=" * 70)

for index, class_name in enumerate(
    CLASS_NAMES
):

    count = np.sum(
        predicted_labels == index
    )

    percentage = (
        count / len(predicted_labels)
    ) * 100

    print(
        f"{class_name:15}: "
        f"{count:3d}/{len(predicted_labels)} "
        f"({percentage:6.2f}%)"
    )

# ============================================================
# SAVE TEXT RESULTS
# ============================================================

with open(
    RESULT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "TruthLens CASIA 2.0 "
        "Combined External Validation\n"
    )

    file.write(
        "======================================\n\n"
    )

    file.write(
        f"Model: {MODEL_PATH}\n"
    )

    file.write(
        f"Real Dataset: {REAL_DIR}\n"
    )

    file.write(
        f"Manipulated Dataset: "
        f"{MANIPULATED_DIR}\n\n"
    )

    file.write(
        f"Real Images: {len(real_images)}\n"
    )

    file.write(
        f"Manipulated Images: "
        f"{len(manipulated_images)}\n"
    )

    file.write(
        f"Total Images: "
        f"{len(true_labels)}\n\n"
    )

    file.write(
        f"Accuracy: "
        f"{accuracy * 100:.4f}%\n"
    )

    file.write(
        f"Macro Precision: "
        f"{precision * 100:.4f}%\n"
    )

    file.write(
        f"Macro Recall: "
        f"{recall * 100:.4f}%\n"
    )

    file.write(
        f"Macro F1 Score: "
        f"{f1 * 100:.4f}%\n"
    )

    file.write(
        f"Average Confidence: "
        f"{average_confidence * 100:.4f}%\n\n"
    )

    file.write(
        "Classification Report:\n"
    )

    file.write(
        report
    )

    file.write(
        "\n\nConfusion Matrix:\n"
    )

    file.write(
        str(cm)
    )

print("\n" + "=" * 70)
print("CASIA COMBINED VALIDATION COMPLETE")
print("=" * 70)

print(
    "\nResults saved to:"
)

print(
    RESULT_FILE
)