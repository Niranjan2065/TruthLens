import os
import numpy as np
import tensorflow as tf
from PIL import Image

# ============================================================
# TRUTHLENS - MOBILE ADAPTED MODEL TEST
# ============================================================

MODEL_PATH = r"D:\TruthLens\Models\truthlens_mobile_adapted.keras"

REAL_IMAGE = r"D:\TruthLens\Mobile_Test\Real\WhatsApp Image 2026-08-20 at 10.00.44 PM.jpeg"

AI_IMAGE = r"D:\TruthLens\Mobile_Test\AI_Generated\Screenshot 2026-08-20 220312.png"

CLASS_NAMES = [
    "AI_Generated",
    "Deepfake",
    "Manipulated",
    "Real"
]

IMAGE_SIZE = (224, 224)


print("=" * 70)
print("TRUTHLENS - MOBILE ADAPTED MODEL TEST")
print("=" * 70)

print()
print("Model:")
print(MODEL_PATH)

print()
print("Checking files...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found.")

if not os.path.exists(REAL_IMAGE):
    raise FileNotFoundError("Real image not found.")

if not os.path.exists(AI_IMAGE):
    raise FileNotFoundError("AI image not found.")

print("All required files found.")

# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING MOBILE-ADAPTED MODEL")
print("=" * 70)

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")
print()
print("Input shape :", model.input_shape)
print("Output shape:", model.output_shape)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image_path, expected_class):

    print()
    print("=" * 70)
    print("EXPECTED:", expected_class.upper())
    print("=" * 70)

    print()
    print("IMAGE:")
    print(os.path.basename(image_path))

    # Load image
    image = Image.open(image_path).convert("RGB")

    print()
    print("Original size:", image.size)

    # Resize exactly like model input
    image = image.resize(IMAGE_SIZE)

    # Convert to numpy
    img_array = np.array(image, dtype=np.float32)

    # IMPORTANT:
    # Mobile-adapted model was trained with pixel values
    # in the 0-255 range.
    img_array = np.expand_dims(img_array, axis=0)

    print("Input shape:", img_array.shape)
    print("Input dtype:", img_array.dtype)
    print("Input min:", img_array.min())
    print("Input max:", img_array.max())

    # Predict
    predictions = model.predict(img_array, verbose=0)[0]

    # Make sure probabilities are normalized
    if not np.isclose(np.sum(predictions), 1.0, atol=0.01):
        predictions = tf.nn.softmax(predictions).numpy()

    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index])

    print()
    print("PREDICTIONS")
    print("-" * 70)

    for class_name, probability in zip(CLASS_NAMES, predictions):
        print(f"{class_name:<15}: {probability * 100:7.2f}%")

    print()
    print("PREDICTED :", predicted_class)
    print("CONFIDENCE:", f"{confidence * 100:.2f}%")

    print()

    if predicted_class == expected_class:
        print("RESULT: CORRECT")
    else:
        print("RESULT: INCORRECT")

    return predicted_class


# ============================================================
# TEST REAL IMAGE
# ============================================================

real_prediction = predict_image(
    REAL_IMAGE,
    "Real"
)


# ============================================================
# TEST AI IMAGE
# ============================================================

ai_prediction = predict_image(
    AI_IMAGE,
    "AI_Generated"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("FINAL MOBILE TEST SUMMARY")
print("=" * 70)

print()
print("Real image")
print("Expected :", "Real")
print("Predicted:", real_prediction)

print()
print("AI-generated image")
print("Expected :", "AI_Generated")
print("Predicted:", ai_prediction)

print()

real_correct = real_prediction == "Real"
ai_correct = ai_prediction == "AI_Generated"

print("Real image test :", "PASS" if real_correct else "FAIL")
print("AI image test   :", "PASS" if ai_correct else "FAIL")

print()

if real_correct and ai_correct:
    print("OVERALL RESULT: MOBILE TEST PASSED")
    print()
    print("The mobile-adapted model correctly classified")
    print("both previously problematic test images.")
else:
    print("OVERALL RESULT: MOBILE TEST NEEDS IMPROVEMENT")
    print()
    print("The model still misclassifies one or more mobile images.")

print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)