import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# TRUTHLENS - TFLITE PREPROCESSING CHECK
# ============================================================

PROJECT_DIR = Path(r"D:\TruthLens")

MODEL_PATH = PROJECT_DIR / "Models" / "truthlens_model.tflite"

REAL_DIR = PROJECT_DIR / "RealWorld_CASIA_Test" / "Real"
MANIPULATED_DIR = PROJECT_DIR / "RealWorld_CASIA_Test" / "Manipulated"

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "AI_Generated",
    "Deepfake",
    "Manipulated",
    "Real"
]

# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("TRUTHLENS - TFLITE PREPROCESSING VERIFICATION")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)

interpreter = tf.lite.Interpreter(
    model_path=str(MODEL_PATH)
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]["index"]
output_index = output_details[0]["index"]

print("\nInput shape:")
print(input_details[0]["shape"])

print("\nInput dtype:")
print(input_details[0]["dtype"])

print("\nInput quantization:")
print(input_details[0]["quantization"])

print("\nOutput shape:")
print(output_details[0]["shape"])

print("\nOutput dtype:")
print(output_details[0]["dtype"])


# ============================================================
# LOAD IMAGES
# ============================================================

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image = np.array(image).astype(np.float32)

    return image


# ============================================================
# GET DATASET
# ============================================================

real_images = [
    p for p in REAL_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]
]

manipulated_images = [
    p for p in MANIPULATED_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]
]

print("\n" + "=" * 70)
print("EXTERNAL TEST DATA")
print("=" * 70)

print(f"\nReal images        : {len(real_images)}")
print(f"Manipulated images : {len(manipulated_images)}")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(image, preprocessing_mode):

    if preprocessing_mode == "0_255":

        processed = image

    elif preprocessing_mode == "0_1":

        processed = image / 255.0

    else:

        raise ValueError("Unknown preprocessing mode")

    processed = np.expand_dims(
        processed,
        axis=0
    ).astype(np.float32)

    interpreter.set_tensor(
        input_index,
        processed
    )

    interpreter.invoke()

    prediction = interpreter.get_tensor(
        output_index
    )

    prediction = prediction[0]

    class_id = int(np.argmax(prediction))

    confidence = float(prediction[class_id])

    return class_id, confidence


# ============================================================
# TEST ONE PREPROCESSING METHOD
# ============================================================

def evaluate(preprocessing_mode):

    y_true = []
    y_pred = []
    confidences = []

    print("\n" + "=" * 70)
    print(f"TESTING PREPROCESSING: {preprocessing_mode}")
    print("=" * 70)

    # --------------------------------------------------------
    # REAL
    # --------------------------------------------------------

    for image_path in real_images:

        image = load_image(image_path)

        predicted_class, confidence = predict(
            image,
            preprocessing_mode
        )

        # Actual class = Real
        y_true.append(3)
        y_pred.append(predicted_class)
        confidences.append(confidence)

    # --------------------------------------------------------
    # MANIPULATED
    # --------------------------------------------------------

    for image_path in manipulated_images:

        image = load_image(image_path)

        predicted_class, confidence = predict(
            image,
            preprocessing_mode
        )

        # Actual class = Manipulated
        y_true.append(2)
        y_pred.append(predicted_class)
        confidences.append(confidence)

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    average_confidence = np.mean(
        confidences
    )

    print("\nResults:")
    print(f"Accuracy            : {accuracy * 100:.2f}%")
    print(f"Average Confidence : {average_confidence * 100:.2f}%")

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=[2, 3],
            target_names=[
                "Manipulated",
                "Real"
            ],
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_true,
            y_pred,
            labels=[2, 3]
        )
    )

    return accuracy


# ============================================================
# RUN BOTH TESTS
# ============================================================

accuracy_255 = evaluate("0_255")

accuracy_01 = evaluate("0_1")


# ============================================================
# COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING COMPARISON")
print("=" * 70)

print(
    f"\n0-255 preprocessing accuracy : "
    f"{accuracy_255 * 100:.2f}%"
)

print(
    f"0-1 preprocessing accuracy   : "
    f"{accuracy_01 * 100:.2f}%"
)

print("\nDifference:")

difference = (
    abs(accuracy_255 - accuracy_01) * 100
)

print(
    f"{difference:.2f} percentage points"
)


# ============================================================
# RECOMMENDATION
# ============================================================

print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)

if accuracy_255 > accuracy_01:

    print(
        "\n0-255 preprocessing performs better."
    )

    print(
        "Use images in the 0-255 range "
        "for TFLite inference."
    )

elif accuracy_01 > accuracy_255:

    print(
        "\n0-1 preprocessing performs better."
    )

    print(
        "Use images divided by 255 "
        "for TFLite inference."
    )

else:

    print(
        "\nBoth preprocessing methods "
        "produced the same accuracy."
    )

print("\nPreprocessing verification completed.")