import tensorflow as tf

MODEL_PATH = r"D:\TruthLens\Models\truthlens_mobile_adapted.keras"

print("=" * 70)
print("TRUTHLENS - MOBILE ADAPTED MODEL INSPECTION")
print("=" * 70)

model = tf.keras.models.load_model(MODEL_PATH)

print("\nMODEL LOADED SUCCESSFULLY")

print("\nInput shape:")
print(model.input_shape)

print("\nOutput shape:")
print(model.output_shape)

print("\n" + "=" * 70)
print("TOP-LEVEL LAYERS")
print("=" * 70)

for i, layer in enumerate(model.layers):
    print(
        f"{i:3d} | "
        f"{layer.name:<35} | "
        f"{type(layer).__name__}"
    )

print("\n" + "=" * 70)
print("MODEL CONFIGURATION")
print("=" * 70)

print("Loss:")
print(model.loss)

print("\nOptimizer:")
print(type(model.optimizer).__name__)

print("\n" + "=" * 70)
print("INPUT / OUTPUT DETAILS")
print("=" * 70)

for tensor in model.inputs:
    print("Input:", tensor)

for tensor in model.outputs:
    print("Output:", tensor)

print("\n" + "=" * 70)
print("LAYER DETAILS")
print("=" * 70)

for layer in model.layers:

    if isinstance(layer, tf.keras.Sequential):
        print("\nSequential layer:", layer.name)

        for j, sublayer in enumerate(layer.layers):
            print(
                f"   {j:2d} | "
                f"{sublayer.name:<30} | "
                f"{type(sublayer).__name__}"
            )

            if hasattr(sublayer, "get_config"):
                config = sublayer.get_config()

                if "factor" in config:
                    print("        factor:", config["factor"])

                if "height_factor" in config:
                    print("        height_factor:",
                          config["height_factor"])

                if "width_factor" in config:
                    print("        width_factor:",
                          config["width_factor"])

                if "horizontal_flip" in config:
                    print("        horizontal_flip:",
                          config["horizontal_flip"])

                if "vertical_flip" in config:
                    print("        vertical_flip:",
                          config["vertical_flip"])

print("\n" + "=" * 70)
print("MODEL INSPECTION COMPLETE")
print("=" * 70)