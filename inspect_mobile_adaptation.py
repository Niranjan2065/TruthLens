import os
import numpy as np
from PIL import Image

BASE = r"D:\TruthLens"

DATASETS = {
    "Original": r"D:\TruthLens\RealWorld_Training",
    "Mobile": r"D:\TruthLens\Mobile_Adapted_Training"
}

CLASSES = [
    "AI_Generated",
    "Deepfake",
    "Manipulated",
    "Real"
]

print("=" * 70)
print("TRUTHLENS - MOBILE ADAPTATION INSPECTION")
print("=" * 70)

for dataset_name, dataset_path in DATASETS.items():

    print()
    print("=" * 70)
    print(dataset_name.upper())
    print("=" * 70)

    for class_name in CLASSES:

        folder = os.path.join(dataset_path, class_name)

        files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
            )
        ]

        print()
        print(class_name)
        print("-" * 50)
        print("Images:", len(files))

        if not files:
            continue

        widths = []
        heights = []
        modes = []

        sample_values = []

        for file in files[:100]:

            try:
                img = Image.open(file)

                widths.append(img.width)
                heights.append(img.height)
                modes.append(img.mode)

                arr = np.array(img)

                sample_values.append(
                    (
                        float(arr.min()),
                        float(arr.max()),
                        float(arr.mean()),
                        float(arr.std())
                    )
                )

            except Exception:
                pass

        print("Width range :", min(widths), "-", max(widths))
        print("Height range:", min(heights), "-", max(heights))

        print("Modes:", set(modes))

        if sample_values:

            mins = [x[0] for x in sample_values]
            maxs = [x[1] for x in sample_values]
            means = [x[2] for x in sample_values]
            stds = [x[3] for x in sample_values]

            print("Pixel minimum :", round(min(mins), 2))
            print("Pixel maximum :", round(max(maxs), 2))
            print("Mean pixel    :", round(np.mean(means), 2))
            print("Pixel std     :", round(np.mean(stds), 2))

print()
print("=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)