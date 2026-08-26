import os
import glob
import numpy as np
from PIL import Image
import tensorflow as tf


# ============================================================
# TRUTHLENS - TFLITE MODEL TEST
# ============================================================

MODEL_PATH = r"D:\TruthLens\Models\truthlens_mobile_adapted.tflite"
TEST_DIR = r"D:\TruthLens\data\val_holdout"

CLASS_NAMES = [
    "AI_Generated",
    "Deepfake",
    "Manipulated",
    "Real"
]

IMAGE_SIZE = (224, 224)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("TRUTHLENS - TFLITE MODEL VERIFICATION")
print("=" * 70)

print()
print("TFLite model:")
print(MODEL_PATH)

print()
print("Test dataset:")
print(TEST_DIR)


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"TFLite model not found:\n{MODEL_PATH}"
    )

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(
        f"Test directory not found:\n{TEST_DIR}"
    )


# ============================================================
# LOAD TFLITE MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING TFLITE MODEL")
print("=" * 70)

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print()
print("Input details:")
print(input_details)

print()
print("Output details:")
print(output_details)


# ============================================================
# GET INPUT / OUTPUT INFORMATION
# ============================================================

input_index = input_details[0]["index"]
output_index = output_details[0]["index"]

input_shape = input_details[0]["shape"]
input_dtype = input_details[0]["dtype"]

output_shape = output_details[0]["shape"]
output_dtype = output_details[0]["dtype"]

print()
print("=" * 70)
print("MODEL INFORMATION")
print("=" * 70)

print(f"Input shape : {input_shape}")
print(f"Input dtype : {input_dtype}")

print(f"Output shape: {output_shape}")
print(f"Output dtype: {output_dtype}")


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path):
    """
    Load and preprocess an image for the TruthLens TFLite model.
    """

    image = Image.open(image_path).convert("RGB")

    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(image)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Convert according to model input type
    if input_dtype == np.float32:
        image_array = image_array.astype(np.float32)

    elif input_dtype == np.uint8:
        image_array = image_array.astype(np.uint8)

    elif input_dtype == np.int8:
        scale, zero_point = input_details[0]["quantization"]

        image_array = image_array.astype(np.float32)

        if scale > 0:
            image_array = image_array / scale + zero_point

        image_array = np.round(image_array).astype(np.int8)

    else:
        image_array = image_array.astype(input_dtype)

    return image_array


# ============================================================
# PREDICT ONE IMAGE
# ============================================================

def predict(image_path):

    image = preprocess_image(image_path)

    interpreter.set_tensor(input_index, image)

    interpreter.invoke()

    output = interpreter.get_tensor(output_index)

    output = np.asarray(output)

    # Remove batch dimension
    probabilities = output[0]

    # If output is quantized, dequantize it
    if output_dtype in [np.int8, np.uint8]:

        scale, zero_point = output_details[0]["quantization"]

        if scale > 0:
            probabilities = (
                probabilities.astype(np.float32) - zero_point
            ) * scale

    # Convert logits to probabilities if necessary
    exp_values = np.exp(
        probabilities - np.max(probabilities)
    )

    probabilities = exp_values / np.sum(exp_values)

    predicted_index = int(np.argmax(probabilities))

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(probabilities[predicted_index])

    return predicted_class, confidence, probabilities


# ============================================================
# FIND TEST IMAGES
# ============================================================

print()
print("=" * 70)
print("SEARCHING TEST DATA")
print("=" * 70)

image_extensions = [
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.bmp",
    "*.webp"
]

image_files = []

for extension in image_extensions:
    image_files.extend(
        glob.glob(
            os.path.join(TEST_DIR, "*", extension)
        )
    )

print()
print(f"Images found: {len(image_files)}")


# ============================================================
# TEST ONE IMAGE FIRST
# ============================================================

if len(image_files) == 0:

    print()
    print("ERROR: No test images found.")
    print()
    print("Expected structure:")
    print(
        r"D:\TruthLens\data\val_holdout\AI_Generated\image.jpg"
    )
    print(
        r"D:\TruthLens\data\val_holdout\Deepfake\image.jpg"
    )
    print(
        r"D:\TruthLens\data\val_holdout\Manipulated\image.jpg"
    )
    print(
        r"D:\TruthLens\data\val_holdout\Real\image.jpg"
    )

    raise SystemExit(1)


# Pick first image
test_image = image_files[0]

print()
print("=" * 70)
print("SINGLE IMAGE TEST")
print("=" * 70)

print()
print("Test image:")
print(test_image)

predicted_class, confidence, probabilities = predict(
    test_image
)

print()
print("Prediction:")
print(f"Class      : {predicted_class}")
print(f"Confidence : {confidence * 100:.2f}%")

print()
print("Class probabilities:")

for class_name, probability in zip(
    CLASS_NAMES,
    probabilities
):
    print(
        f"{class_name:<15}: {probability * 100:.2f}%"
    )


# ============================================================
# TEST ALL VALIDATION IMAGES
# ============================================================

print()
print("=" * 70)
print("FULL VALIDATION TEST")
print("=" * 70)

confusion_matrix = np.zeros(
    (len(CLASS_NAMES), len(CLASS_NAMES)),
    dtype=np.int32
)

total = 0
correct = 0

for counter, image_path in enumerate(image_files, start=1):

    # Determine true class from folder name
    true_class = os.path.basename(
        os.path.dirname(image_path)
    )

    if true_class not in CLASS_NAMES:
        continue

    true_index = CLASS_NAMES.index(true_class)

    try:

        predicted_class, confidence, probabilities = predict(
            image_path
        )

        predicted_index = CLASS_NAMES.index(
            predicted_class
        )

        confusion_matrix[
            true_index,
            predicted_index
        ] += 1

        total += 1

        if true_index == predicted_index:
            correct += 1

    except Exception as error:

        print()
        print("ERROR processing:")
        print(image_path)
        print(error)

    # Progress
    if counter % 100 == 0:

        print(
            f"Processed {counter}/{len(image_files)}"
        )


# ============================================================
# ACCURACY
# ============================================================

accuracy = 0.0

if total > 0:
    accuracy = correct / total


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print()
print("Rows = True class")
print("Columns = Predicted class")

print()
print("Class order:")
print(CLASS_NAMES)

print()

print(confusion_matrix)


# ============================================================
# PER-CLASS RESULTS
# ============================================================

print()
print("=" * 70)
print("PER-CLASS RESULTS")
print("=" * 70)

for i, class_name in enumerate(CLASS_NAMES):

    true_positive = confusion_matrix[i, i]

    actual = np.sum(confusion_matrix[i, :])

    predicted = np.sum(confusion_matrix[:, i])

    recall = (
        true_positive / actual
        if actual > 0
        else 0
    )

    precision = (
        true_positive / predicted
        if predicted > 0
        else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    print()
    print(class_name)

    print(
        f"  Samples   : {actual}"
    )

    print(
        f"  Correct   : {true_positive}"
    )

    print(
        f"  Precision : {precision * 100:.2f}%"
    )

    print(
        f"  Recall    : {recall * 100:.2f}%"
    )

    print(
        f"  F1-score  : {f1 * 100:.2f}%"
    )


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 70)
print("FINAL TFLITE TEST RESULT")
print("=" * 70)

print()
print(f"Total images : {total}")
print(f"Correct      : {correct}")
print(f"Incorrect    : {total - correct}")

print()
print(
    f"TFLite Accuracy: {accuracy * 100:.2f}%"
)

print()

print("=" * 70)
print("TFLITE MODEL VERIFICATION COMPLETE")
print("=" * 70)

print()
print("If the model runs successfully,")
print("the TFLite file is ready for Android integration.")

print()