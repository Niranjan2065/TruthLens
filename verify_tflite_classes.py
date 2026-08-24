import os
import numpy as np
import tensorflow as tf
from PIL import Image

# ============================================================
# TruthLens - TFLite Class Order Verification
# ============================================================

MODEL_PATH = r"D:\TruthLens\Models\truthlens_model.tflite"

# Change this only if your test folder has a different location
TEST_DIR = r"D:\TruthLens\Processed_Dataset"
# These are the TRUE folder/class names in your dataset
CLASS_NAMES = [
    "Real",
    "AI_Generated",
    "Manipulated",
    "Deepfake"
]

IMAGE_SIZE = (224, 224)
MAX_IMAGES_PER_CLASS = 100


# ============================================================
# 1. Check model
# ============================================================

print("=" * 60)
print("TRUTHLENS TFLITE CLASS ORDER VERIFICATION")
print("=" * 60)

if not os.path.exists(MODEL_PATH):
    print("\nERROR: Model not found:")
    print(MODEL_PATH)
    exit()

print("\nModel found:")
print(MODEL_PATH)


# ============================================================
# 2. Load TFLite model
# ============================================================

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nInput information:")
print(input_details)

print("\nOutput information:")
print(output_details)


# ============================================================
# 3. Check input shape
# ============================================================

input_shape = input_details[0]["shape"]

print("\nInput shape:")
print(input_shape)

if input_shape[1] != 224 or input_shape[2] != 224:
    print("\nWARNING:")
    print("Model input is not 224 x 224.")
    print("Please check your original preprocessing.")


# ============================================================
# 4. Check output shape
# ============================================================

output_shape = output_details[0]["shape"]

print("\nOutput shape:")
print(output_shape)

num_outputs = output_shape[-1]

print("\nNumber of output classes:", num_outputs)

if num_outputs != 4:
    print("\nWARNING: Expected 4 output classes.")
    exit()


# ============================================================
# 5. Image preprocessing
# ============================================================

def preprocess_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = image.resize(IMAGE_SIZE)

    image = np.array(image).astype(np.float32)

    # EfficientNet preprocessing
    image = tf.keras.applications.efficientnet.preprocess_input(image)

    image = np.expand_dims(image, axis=0)

    return image


# ============================================================
# 6. Run one image
# ============================================================

def predict(image_path):

    image = preprocess_image(image_path)

    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]

    # Handle different TFLite input types
    input_dtype = input_details[0]["dtype"]

    if input_dtype == np.uint8:

        scale, zero_point = input_details[0]["quantization"]

        if scale != 0:
            image = image / scale + zero_point

        image = np.clip(image, 0, 255).astype(np.uint8)

    elif input_dtype == np.float32:

        image = image.astype(np.float32)

    else:

        image = image.astype(input_dtype)

    interpreter.set_tensor(input_index, image)

    interpreter.invoke()

    output = interpreter.get_tensor(output_index)[0]

    # Dequantize output if necessary
    output_dtype = output_details[0]["dtype"]

    if output_dtype == np.uint8:

        scale, zero_point = output_details[0]["quantization"]

        if scale != 0:
            output = (output.astype(np.float32) - zero_point) * scale

    return output


# ============================================================
# 7. Find images
# ============================================================

print("\n" + "=" * 60)
print("SEARCHING TEST DATASET")
print("=" * 60)

if not os.path.exists(TEST_DIR):

    print("\nERROR: Test directory not found:")
    print(TEST_DIR)

    print("\nPlease check your Processed_Dataset folder.")

    exit()


# ============================================================
# 8. Test every class
# ============================================================

all_results = {}

for true_class in CLASS_NAMES:

    class_dir = os.path.join(TEST_DIR, true_class)

    print("\n")
    print("-" * 60)
    print("TRUE CLASS:", true_class)
    print("-" * 60)

    if not os.path.exists(class_dir):

        print("WARNING: Folder not found:")
        print(class_dir)

        continue

    files = []

    for filename in os.listdir(class_dir):

        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        ):
            files.append(os.path.join(class_dir, filename))

    files = files[:MAX_IMAGES_PER_CLASS]

    print("Images tested:", len(files))

    if len(files) == 0:

        print("No images found.")
        continue

    predictions = []

    for image_path in files:

        try:

            output = predict(image_path)

            predictions.append(output)

        except Exception as e:

            print("Error:", os.path.basename(image_path))
            print(e)

    if len(predictions) == 0:

        continue

    predictions = np.array(predictions)

    average_prediction = np.mean(predictions, axis=0)

    predicted_index = int(np.argmax(average_prediction))

    confidence = float(average_prediction[predicted_index])

    all_results[true_class] = (
        predicted_index,
        average_prediction
    )

    print("\nAverage model output:")

    for i, value in enumerate(average_prediction):

        print(
            f"Output index {i}: "
            f"{value:.6f}"
        )

    print("\nDominant output index:", predicted_index)

    print(
        f"Dominant confidence: "
        f"{confidence * 100:.2f}%"
    )


# ============================================================
# 9. Build class mapping
# ============================================================

print("\n")
print("=" * 60)
print("RESULT")
print("=" * 60)

mapping = {}

for true_class, result in all_results.items():

    predicted_index = result[0]

    mapping[predicted_index] = true_class


print("\nDetected model output mapping:\n")

for index in range(4):

    if index in mapping:

        print(
            f"{index} -> {mapping[index]}"
        )

    else:

        print(
            f"{index} -> UNKNOWN"
        )


# ============================================================
# 10. Generate labels.txt order
# ============================================================

print("\n")
print("=" * 60)
print("ANDROID labels.txt ORDER")
print("=" * 60)

print()

for index in range(4):

    if index in mapping:

        print(mapping[index])

    else:

        print("UNKNOWN")


# ============================================================
# 11. Save automatically
# ============================================================

labels_path = r"D:\TruthLens\verified_labels.txt"

with open(labels_path, "w", encoding="utf-8") as file:

    for index in range(4):

        if index in mapping:

            file.write(mapping[index] + "\n")
        else:

            file.write("UNKNOWN\n")


print("\nVerified labels saved to:")

print(labels_path)

print("\nVerification complete.")