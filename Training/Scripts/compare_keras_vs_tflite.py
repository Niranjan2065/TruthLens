import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image

# ==========================================================
# CONFIGURATION
# ==========================================================

KERAS_MODEL = r"D:\TruthLens\Models\truthlens_efficientnetb0_finetuned.keras"
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
# LOAD MODELS
# ==========================================================

print("=" * 70)
print("COMPARING KERAS VS TFLITE")
print("=" * 70)

print("\nLoading Keras model...")

keras_model = tf.keras.models.load_model(KERAS_MODEL)

print("Keras model loaded.")

print("\nLoading TFLite model...")

interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("TFLite model loaded.")

# ==========================================================
# PREPROCESS
# ==========================================================

def preprocess(image_path):

    img = Image.open(image_path).convert("RGB")

    img = img.resize(IMG_SIZE)

    # IMPORTANT:
    # Do NOT normalize by dividing by 255.
    img = np.array(img, dtype=np.float32)

    img = np.expand_dims(img, axis=0)

    return img

# ==========================================================
# KERAS
# ==========================================================

def keras_predict(img):

    pred = keras_model.predict(img, verbose=0)

    cls = np.argmax(pred)

    conf = pred[0][cls]

    return cls, conf, pred

# ==========================================================
# TFLITE
# ==========================================================

def tflite_predict(img):

    interpreter.set_tensor(
        input_details[0]["index"],
        img
    )

    interpreter.invoke()

    pred = interpreter.get_tensor(
        output_details[0]["index"]
    )

    cls = np.argmax(pred)

    conf = pred[0][cls]

    return cls, conf, pred

# ==========================================================
# COMPARE
# ==========================================================

print("\n")

for class_name in CLASS_NAMES:

    print("=" * 70)
    print("CLASS :", class_name)
    print("=" * 70)

    folder = TEST_DIR / class_name

    images = sorted(folder.glob("*.jpg"))[:3]

    for img_path in images:

        img = preprocess(img_path)

        k_cls, k_conf, _ = keras_predict(img)

        t_cls, t_conf, _ = tflite_predict(img)

        print(f"\nImage : {img_path.name}")

        print(
            f"Keras  : {CLASS_NAMES[k_cls]:15} {k_conf*100:6.2f}%"
        )

        print(
            f"TFLite : {CLASS_NAMES[t_cls]:15} {t_conf*100:6.2f}%"
        )

        if k_cls == t_cls:
            print("Result : MATCH")
        else:
            print("Result : DIFFERENT")

print("\n")
print("=" * 70)
print("COMPARISON COMPLETED")
print("=" * 70)