import tensorflow as tf
import numpy as np
from PIL import Image
import os

MODEL_PATH = r"D:\TruthLens\Models\truthlens_model.tflite"

IMAGE_SIZE = 224

LABELS = [
    "AI_Generated",
    "Deepfake",
    "Manipulated",
    "Real"
]


def load_image(image_path):

    image = Image.open(image_path).convert("RGB")

    print("\nImage:", os.path.basename(image_path))
    print("Original size:", image.size)

    # Same resize used by Android
    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    image = np.asarray(
        image,
        dtype=np.float32
    )

    # IMPORTANT:
    # Same preprocessing currently used in Android
    image = image.reshape(
        1,
        IMAGE_SIZE,
        IMAGE_SIZE,
        3
    )

    return image


# --------------------------------------------------
# Load TFLite model
# --------------------------------------------------

interpreter = tf.lite.Interpreter(
    model_path=MODEL_PATH
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\n========================================")
print("TRUTHLENS TFLITE TEST")
print("========================================")

print("Input shape:",
      input_details[0]["shape"])

print("Input dtype:",
      input_details[0]["dtype"])

print("Output shape:",
      output_details[0]["shape"])

print("========================================")


# --------------------------------------------------
# Prediction function
# --------------------------------------------------

def predict(image_path):

    image = load_image(image_path)

    interpreter.set_tensor(
        input_details[0]["index"],
        image
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_details[0]["index"]
    )[0]

    best_index = int(
        np.argmax(output)
    )

    print("\n----------------------------------------")
    print("FILE:", os.path.basename(image_path))
    print("----------------------------------------")

    for i, probability in enumerate(output):

        print(
            f"{LABELS[i]:15s}: "
            f"{probability * 100:.2f}%"
        )

    print("----------------------------------------")

    print(
        "PREDICTION:",
        LABELS[best_index]
    )

    print(
        "CONFIDENCE:",
        f"{output[best_index] * 100:.2f}%"
    )

    return output


# --------------------------------------------------
# Test images
# --------------------------------------------------

test_folder = r"D:\TruthLens\Test_Images"

for filename in os.listdir(test_folder):

    path = os.path.join(
        test_folder,
        filename
    )

    if filename.lower().endswith(
        (".jpg", ".jpeg", ".png", ".webp")
    ):

        predict(path)


print("\n========================================")
print("TEST COMPLETE")
print("========================================")