import os
from PIL import Image

DATASET = r"D:\TruthLens\RealWorld_Training"

CLASSES = [
    "AI_Generated",
    "Deepfake",
    "Manipulated",
    "Real"
]

print("=" * 70)
print("TRUTHLENS - REAL-WORLD TRAINING DATA INSPECTION")
print("=" * 70)

total = 0

for cls in CLASSES:

    folder = os.path.join(DATASET, cls)

    print("\n" + "-" * 70)
    print(f"CLASS: {cls}")
    print("-" * 70)

    if not os.path.exists(folder):
        print("ERROR: Folder not found")
        continue

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp", ".bmp")
        )
    ]

    valid = 0
    invalid = 0

    widths = []
    heights = []

    for filename in files:

        path = os.path.join(folder, filename)

        try:
            with Image.open(path) as img:

                img.verify()

            with Image.open(path) as img:
                widths.append(img.width)
                heights.append(img.height)

            valid += 1

        except Exception:
            invalid += 1

    total += valid

    print(f"Total files      : {len(files)}")
    print(f"Valid images     : {valid}")
    print(f"Invalid images   : {invalid}")

    if widths:
        print(f"Minimum width    : {min(widths)}")
        print(f"Maximum width    : {max(widths)}")
        print(f"Minimum height   : {min(heights)}")
        print(f"Maximum height   : {max(heights)}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"Total valid images: {total}")

print("\nExpected:")
print("AI_Generated : 500")
print("Deepfake     : 500")
print("Manipulated  : 500")
print("Real         : 500")
print("Total        : 2000")

print("=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)