import os
import numpy as np
import tensorflow as tf
from PIL import Image

# ============================================================
# TRUTHLENS - EXACT MOBILE IMAGE KERAS TEST
# ============================================================

MODEL_PATH = r"D:\TruthLens\Models\truthlens_efficientnetb0_finetuned_final.keras"

REAL_IMAGE = r"D:\TruthLens\Mobile_Test\Real\WhatsApp Image 2026-08-20 at 10.00.44 PM.jpeg"

AI_IMAGE = r"D:\TruthLens\Mobile_Test\AI_Generated\Screenshot 2026-08-20 220312.png"

CLASS_NAMES = [
    "AI_Generated",
    "Deepfake",
    "Manipulated",
    "Real"
]

IMG_SIZE = (224, 224)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("TRUTHLENS - EXACT MOBILE IMAGE KERAS TEST")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)


# ============================================================
# CHECK IMAGE FILES
# ============================================================

print("\nChecking image files...")

if os.path.exists(REAL_IMAGE):
    print("FOUND:", REAL_IMAGE)
else:
    print("ERROR - REAL IMAGE NOT FOUND:")
    print(REAL_IMAGE)

if os.path.exists(AI_IMAGE):
    print("FOUND:", AI_IMAGE)
else:
    print("ERROR - AI IMAGE NOT FOUND:")
    print(AI_IMAGE)


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING KERAS MODEL")
print("=" * 70)

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

print("\nInput shape:")
print(model.input_shape)

print("\nOutput shape:")
print(model.output_shape)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image_path, expected_class):

    print("\n")
    print("=" * 70)
    print("EXPECTED:", expected_class.upper())
    print("=" * 70)

    print("\n" + "-" * 70)
    print("IMAGE:")
    print(os.path.basename(image_path))
    print("-" * 70)

    # --------------------------------------------------------
    # Check image
    # --------------------------------------------------------

    if not os.path.exists(image_path):
        print("\nERROR: Image not found.")
        print(image_path)
        return

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    img = Image.open(image_path).convert("RGB")

    print("\nOriginal size:", img.size)

    # --------------------------------------------------------
    # Resize to model input size
    # --------------------------------------------------------

    img = img.resize(IMG_SIZE)

    # --------------------------------------------------------
    # Convert image to NumPy array
    # --------------------------------------------------------

    img_array = np.array(img)

    # --------------------------------------------------------
    # Convert to float32
    #
    # IMPORTANT:
    # DO NOT divide by 255 here.
    #
    # Pixel range remains:
    # 0 - 255
    # --------------------------------------------------------

    img_array = img_array.astype("float32")

    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    image_array = np.expand_dims(img_array, axis=0)

    # --------------------------------------------------------
    # Verify input
    # --------------------------------------------------------

    print("Input shape:", image_array.shape)
    print("Input dtype:", image_array.dtype)
    print("Input min:", image_array.min())
    print("Input max:", image_array.max())

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    predictions = model.predict(
        image_array,
        verbose=0
    )

    predictions = predictions[0]

    # --------------------------------------------------------
    # Check output
    # --------------------------------------------------------

    print("\nRaw Keras output:")
    print(predictions)

    print("\nOutput sum:")
    print(np.sum(predictions))

    # --------------------------------------------------------
    # Determine whether output is already probabilities
    # --------------------------------------------------------

    output_sum = np.sum(predictions)

    if (
        np.all(predictions >= 0)
        and
        np.all(predictions <= 1)
        and
        abs(output_sum - 1.0) < 0.01
    ):

        print("\nOutput type: ALREADY NORMALIZED PROBABILITIES")

        probabilities = predictions

    else:

        print("\nOutput type: LOGITS")
        print("Applying softmax...")

        probabilities = tf.nn.softmax(predictions).numpy()

    # --------------------------------------------------------
    # Print probabilities
    # --------------------------------------------------------

    print("\nProbabilities:")

    for class_name, probability in zip(
        CLASS_NAMES,
        probabilities
    ):

        print(
            f"{class_name:<15}: "
            f"{probability * 100:7.2f}%"
        )

    # --------------------------------------------------------
    # Get predicted class
    # --------------------------------------------------------

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(
        probabilities[predicted_index]
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\nPREDICTED:", predicted_class)

    print(
        f"CONFIDENCE: {confidence * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Expected vs predicted
    # --------------------------------------------------------

    if predicted_class.lower() == expected_class.lower():

        print("\nRESULT: CORRECT")

    else:

        print("\nRESULT: INCORRECT")


# ============================================================
# TEST REAL IMAGE
# ============================================================

predict_image(
    REAL_IMAGE,
    "Real"
)


# ============================================================
# TEST AI GENERATED IMAGE
# ============================================================

predict_image(
    AI_IMAGE,
    "AI_Generated"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("KERAS MOBILE IMAGE TEST COMPLETE")
print("=" * 70)