import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image

# ==========================================================
# CONFIGURATION
# ==========================================================

TFLITE_MODEL = r"D:\TruthLens\Models\truthlens_model.tflite"

TEST_DIR = Path(r"D:\TruthLens\Final_Dataset\Test")

CLASS_NAMES = [
    "AI_Generated",
    "Deepfake",
    "Manipulated",
    "Real"
]

IMG_SIZE = (224, 224)

# ==========================================================
# LOAD INTERPRETER
# ==========================================================

print("=" * 70)
print("VERIFYING TFLITE MODEL")
print("=" * 70)

interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nModel loaded successfully.")
print("Input Shape :", input_details[0]["shape"])
print("Output Shape:", output_details[0]["shape"])

# ==========================================================
# PREPROCESS
# ==========================================================

def preprocess(image_path):

    img = Image.open(image_path).convert("RGB")

    img = img.resize(IMG_SIZE)

    img = np.array(img).astype(np.float32)

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    return img

# ==========================================================
# PREDICT
# ==========================================================

def predict(image):

    interpreter.set_tensor(
        input_details[0]["index"],
        image
    )

    interpreter.invoke()

    prediction = interpreter.get_tensor(
        output_details[0]["index"]
    )

    class_id = np.argmax(prediction)

    confidence = prediction[0][class_id]

    return class_id, confidence

# ==========================================================
# VERIFY
# ==========================================================

print("\nRunning inference...\n")

correct = 0
total = 0

for class_name in CLASS_NAMES:

    folder = TEST_DIR / class_name

    print("=" * 60)
    print("CLASS :", class_name)
    print("=" * 60)

    images = list(folder.glob("*.jpg"))[:5]

    for img_path in images:

        image = preprocess(img_path)

        pred, conf = predict(image)

        predicted = CLASS_NAMES[pred]

        if predicted == class_name:
            correct += 1

        total += 1

        print(f"{img_path.name:30} -> {predicted:15} {conf*100:6.2f}%")

print("\n" + "=" * 70)

print("Verification Completed")

print("=" * 70)

print(f"Images Tested : {total}")

print(f"Correct       : {correct}")

print(f"Accuracy      : {(correct/total)*100:.2f}%")