from pathlib import Path
from tensorflow import keras

PROJECT_DIR = Path(r"D:\TruthLens")

MODEL_PATH = PROJECT_DIR / "Models" / "truthlens_efficientnetb0_finetuned.keras"

print("=" * 70)
print("TRUTHLENS MODEL INSPECTION")
print("=" * 70)

model = keras.models.load_model(MODEL_PATH)

print("\nModel Loaded Successfully\n")

print("Top-Level Layers")
print("-" * 70)

for i, layer in enumerate(model.layers):
    print(f"{i:2d} | {layer.name:35} | {layer.__class__.__name__}")

print("\n")

base_model = None

for layer in model.layers:
    if "efficientnet" in layer.name.lower():
        base_model = layer
        break

if base_model is None:
    print("EfficientNet not found!")
    exit()

print("=" * 70)
print("LAST 30 LAYERS OF EFFICIENTNET")
print("=" * 70)

for layer in base_model.layers[-30:]:
    print(layer.name, layer.output.shape)