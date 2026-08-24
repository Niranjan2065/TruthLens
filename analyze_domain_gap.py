from pathlib import Path
from PIL import Image
import numpy as np
import statistics

# ============================================================
# TRUTHLENS - DOMAIN GAP ANALYSIS
# ============================================================

PROJECT_DIR = Path(r"D:\TruthLens")

INTERNAL_REAL = (
    PROJECT_DIR
    / "Final_Dataset"
    / "Test"
    / "Real"
)

CASIA_REAL = (
    PROJECT_DIR
    / "RealWorld_CASIA_Test"
    / "Real"
)

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


def analyze_dataset(dataset_dir):

    widths = []
    heights = []
    brightness = []
    contrast = []
    red_values = []
    green_values = []
    blue_values = []
    file_sizes = []

    files = [
        f for f in dataset_dir.iterdir()
        if f.is_file()
        and f.suffix.lower() in VALID_EXTENSIONS
    ]

    for file in files:

        try:

            image = Image.open(file).convert("RGB")

            array = np.asarray(image).astype(np.float32)

            widths.append(image.width)
            heights.append(image.height)

            brightness.append(
                np.mean(array)
            )

            contrast.append(
                np.std(array)
            )

            red_values.append(
                np.mean(array[:, :, 0])
            )

            green_values.append(
                np.mean(array[:, :, 1])
            )

            blue_values.append(
                np.mean(array[:, :, 2])
            )

            file_sizes.append(
                file.stat().st_size / 1024
            )

        except Exception as e:

            print(
                f"Skipping {file.name}: {e}"
            )

    return {
        "count": len(files),
        "width_mean": statistics.mean(widths),
        "height_mean": statistics.mean(heights),
        "width_min": min(widths),
        "width_max": max(widths),
        "height_min": min(heights),
        "height_max": max(heights),
        "brightness_mean": statistics.mean(brightness),
        "contrast_mean": statistics.mean(contrast),
        "red_mean": statistics.mean(red_values),
        "green_mean": statistics.mean(green_values),
        "blue_mean": statistics.mean(blue_values),
        "file_size_mean": statistics.mean(file_sizes)
    }


print("=" * 70)
print("TRUTHLENS - DOMAIN GAP ANALYSIS")
print("=" * 70)

print("\nInternal Real dataset:")
print(INTERNAL_REAL)

print("\nCASIA Real dataset:")
print(CASIA_REAL)

# ============================================================
# ANALYZE
# ============================================================

print("\n" + "=" * 70)
print("ANALYZING INTERNAL REAL DATASET")
print("=" * 70)

internal = analyze_dataset(
    INTERNAL_REAL
)

print("\n" + "=" * 70)
print("ANALYZING CASIA REAL DATASET")
print("=" * 70)

casia = analyze_dataset(
    CASIA_REAL
)

# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("DOMAIN GAP RESULTS")
print("=" * 70)

print("\nMetric                         Internal Real       CASIA Real")
print("-" * 70)

print(
    f"Images                        "
    f"{internal['count']:>8}           "
    f"{casia['count']:>8}"
)

print(
    f"Average width                 "
    f"{internal['width_mean']:>8.1f}           "
    f"{casia['width_mean']:>8.1f}"
)

print(
    f"Average height                "
    f"{internal['height_mean']:>8.1f}           "
    f"{casia['height_mean']:>8.1f}"
)

print(
    f"Min width                     "
    f"{internal['width_min']:>8}           "
    f"{casia['width_min']:>8}"
)

print(
    f"Max width                     "
    f"{internal['width_max']:>8}           "
    f"{casia['width_max']:>8}"
)

print(
    f"Min height                    "
    f"{internal['height_min']:>8}           "
    f"{casia['height_min']:>8}"
)

print(
    f"Max height                    "
    f"{internal['height_max']:>8}           "
    f"{casia['height_max']:>8}"
)

print(
    f"Brightness                    "
    f"{internal['brightness_mean']:>8.2f}           "
    f"{casia['brightness_mean']:>8.2f}"
)

print(
    f"Contrast                      "
    f"{internal['contrast_mean']:>8.2f}           "
    f"{casia['contrast_mean']:>8.2f}"
)

print(
    f"Red channel mean              "
    f"{internal['red_mean']:>8.2f}           "
    f"{casia['red_mean']:>8.2f}"
)

print(
    f"Green channel mean            "
    f"{internal['green_mean']:>8.2f}           "
    f"{casia['green_mean']:>8.2f}"
)

print(
    f"Blue channel mean             "
    f"{internal['blue_mean']:>8.2f}           "
    f"{casia['blue_mean']:>8.2f}"
)

print(
    f"Average file size (KB)        "
    f"{internal['file_size_mean']:>8.2f}           "
    f"{casia['file_size_mean']:>8.2f}"
)

# ============================================================
# DIFFERENCES
# ============================================================

print("\n" + "=" * 70)
print("ABSOLUTE DIFFERENCES")
print("=" * 70)

print(
    f"\nBrightness difference : "
    f"{abs(internal['brightness_mean'] - casia['brightness_mean']):.2f}"
)

print(
    f"Contrast difference   : "
    f"{abs(internal['contrast_mean'] - casia['contrast_mean']):.2f}"
)

print(
    f"Red difference        : "
    f"{abs(internal['red_mean'] - casia['red_mean']):.2f}"
)

print(
    f"Green difference      : "
    f"{abs(internal['green_mean'] - casia['green_mean']):.2f}"
)

print(
    f"Blue difference       : "
    f"{abs(internal['blue_mean'] - casia['blue_mean']):.2f}"
)

print("\n" + "=" * 70)
print("DOMAIN GAP ANALYSIS COMPLETE")
print("=" * 70)