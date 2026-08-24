import os
import numpy as np
from PIL import Image
import tensorflow as tf

# ============================================================
# TRUTHLENS - TFLITE vs KERAS VERIFICATION
# ============================================================

KERAS_MODEL = r"D:\TruthLens\Models\truthlens_efficientnetb0_finetuned_final.keras"
TFLITE_MODEL = r"D:\TruthLens\Models\truthlens_model.tflite"

REAL_IMAGE = r"D:\TruthLens\Mobile_Test\Real\WhatsApp Image 2026-08-20 at 10.00.44 PM.jpeg"
AI_IMAGE = r"D:\TruthLens\Mobile_Test\AI_Generated\Screenshot 2026-08-20 220312.png"

CLASS_NAMES = [
    "AI_Generated",
    "Deepfake",
    "Manipulated",
    "Real"
]

IMAGE_SIZE = (224, 224)


def load_image(path):
    """Load image using the same basic preprocessing for both models."""

    image = Image.open(path).convert("RGB")

    original_size = image.size

    image = image.resize(IMAGE_SIZE)

    # IMPORTANT:
    # Keep values in 0-255 because our Keras test showed
    # the model expects this range.
    array = np.asarray(image, dtype=np.float32)

    array = np.expand_dims(array, axis=0)

    return array, original_size


def print_predictions(title, output):

    output = np.asarray(output).flatten()

    print()
    print(title)
    print("-" * 60)

    for i, value in enumerate(output):
        print(f"{CLASS_NAMES[i]:15s}: {value * 100:8.4f}%")

    predicted_index = int(np.argmax(output))
    confidence = float(output[predicted_index])

    print()
    print("Predicted      :", CLASS_NAMES[predicted_index])
    print(f"Confidence     : {confidence * 100:.4f}%")

    return predicted_index, confidence


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("TRUTHLENS - TFLITE vs KERAS VERIFICATION")
print("=" * 70)

print()
print("Keras model:")
print(KERAS_MODEL)

print()
print("TFLite model:")
print(TFLITE_MODEL)


# ============================================================
# CHECK FILES
# ============================================================

for path in [KERAS_MODEL, TFLITE_MODEL, REAL_IMAGE, AI_IMAGE]:

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found:\n{path}")

print()
print("All required files found.")


# ============================================================
# LOAD KERAS
# ============================================================

print()
print("=" * 70)
print("LOADING KERAS MODEL")
print("=" * 70)

keras_model = tf.keras.models.load_model(
    KERAS_MODEL,
    compile=False
)

print("Keras model loaded.")

print("Input shape :", keras_model.input_shape)
print("Output shape:", keras_model.output_shape)


# ============================================================
# LOAD TFLITE
# ============================================================

print()
print("=" * 70)
print("LOADING TFLITE MODEL")
print("=" * 70)

interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("TFLite model loaded.")

print()
print("Input details:")
print(input_details)

print()
print("Output details:")
print(output_details)


input_index = input_details[0]["index"]
output_index = output_details[0]["index"]

input_shape = input_details[0]["shape"]
input_dtype = input_details[0]["dtype"]

print()
print("TFLite input shape :", input_shape)
print("TFLite input dtype :", input_dtype)


# ============================================================
# VERIFY INPUT COMPATIBILITY
# ============================================================

if list(input_shape) != [1, 224, 224, 3]:

    print()
    print("WARNING:")
    print("TFLite input shape is not [1, 224, 224, 3].")

if input_dtype != np.float32:

    print()
    print("WARNING:")
    print("TFLite input dtype is not float32.")


# ============================================================
# FUNCTION TO TEST ONE IMAGE
# ============================================================

def test_image(image_path, expected_label):

    print()
    print("=" * 70)
    print("IMAGE TEST")
    print("=" * 70)

    print()
    print("Image:")
    print(image_path)

    print()
    print("Expected:")
    print(expected_label)

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image_array, original_size = load_image(image_path)

    print()
    print("Original size :", original_size)
    print("Input shape   :", image_array.shape)
    print("Input dtype   :", image_array.dtype)
    print("Input min     :", image_array.min())
    print("Input max     :", image_array.max())

    # --------------------------------------------------------
    # KERAS
    # --------------------------------------------------------

    keras_output = keras_model.predict(
        image_array,
        verbose=0
    )[0]

    keras_pred, keras_conf = print_predictions(
        "KERAS OUTPUT",
        keras_output
    )

    # --------------------------------------------------------
    # TFLITE
    # --------------------------------------------------------

    tflite_input = image_array.astype(np.float32)

    interpreter.set_tensor(
        input_index,
        tflite_input
    )

    interpreter.invoke()

    tflite_output = interpreter.get_tensor(
        output_index
    )[0]

    tflite_pred, tflite_conf = print_predictions(
        "TFLITE OUTPUT",
        tflite_output
    )

    # --------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)

    print()
    print("Expected        :", expected_label)
    print("Keras prediction:", CLASS_NAMES[keras_pred])
    print("TFLite prediction:", CLASS_NAMES[tflite_pred])

    print()
    print(
        f"Keras confidence : {keras_conf * 100:.4f}%"
    )

    print(
        f"TFLite confidence: {tflite_conf * 100:.4f}%"
    )

    # Difference between output probabilities
    difference = np.abs(
        keras_output - tflite_output
    )

    max_difference = float(np.max(difference))
    mean_difference = float(np.mean(difference))

    print()
    print(
        f"Maximum probability difference : "
        f"{max_difference:.8f}"
    )

    print(
        f"Mean probability difference    : "
        f"{mean_difference:.8f}"
    )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    same_prediction = keras_pred == tflite_pred

    close_output = max_difference < 0.01

    print()

    if same_prediction and close_output:

        print("STATUS: PASS")
        print(
            "TFLite behavior matches Keras closely."
        )

    elif same_prediction:

        print("STATUS: PARTIAL")
        print(
            "Prediction is the same, but probability "
            "values differ."
        )

    else:

        print("STATUS: FAIL")
        print(
            "TFLite prediction differs from Keras."
        )

    return {
        "keras_pred": keras_pred,
        "tflite_pred": tflite_pred,
        "keras_output": keras_output,
        "tflite_output": tflite_output,
        "max_difference": max_difference,
        "mean_difference": mean_difference
    }


# ============================================================
# TEST REAL IMAGE
# ============================================================

real_result = test_image(
    REAL_IMAGE,
    "Real"
)


# ============================================================
# TEST AI GENERATED IMAGE
# ============================================================

ai_result = test_image(
    AI_IMAGE,
    "AI_Generated"
)


# ============================================================
# FINAL VERIFICATION
# ============================================================

print()
print("=" * 70)
print("FINAL TFLITE VERIFICATION")
print("=" * 70)

real_match = (
    real_result["keras_pred"]
    ==
    real_result["tflite_pred"]
)

ai_match = (
    ai_result["keras_pred"]
    ==
    ai_result["tflite_pred"]
)

print()
print(
    "Real image Keras/TFLite match :",
    "YES" if real_match else "NO"
)

print(
    "AI image Keras/TFLite match   :",
    "YES" if ai_match else "NO"
)

print()

if real_match and ai_match:

    print("OVERALL STATUS: PASS")
    print()
    print(
        "TFLite conversion is behaving consistently "
        "with the Keras model for these test images."
    )

else:

    print("OVERALL STATUS: FAIL")
    print()
    print(
        "TFLite and Keras are producing different "
        "predictions for at least one test image."
    )

print()
print("=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)