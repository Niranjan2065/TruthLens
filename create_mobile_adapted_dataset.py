import os
import random
import shutil
import cv2
import numpy as np
from PIL import Image

# ============================================================
# TRUTHLENS - MOBILE ADAPTED DATASET CREATION
# ============================================================

SOURCE_DIR = r"D:\TruthLens\RealWorld_Training"
OUTPUT_DIR = r"D:\TruthLens\Mobile_Adapted_Training"

CLASSES = [
    "AI_Generated",
    "Deepfake",
    "Manipulated",
    "Real"
]

IMAGES_PER_CLASS = 500

random.seed(42)
np.random.seed(42)


def jpeg_compression(image, quality):
    """
    Simulate WhatsApp / social-media JPEG compression.
    """

    encode_param = [
        int(cv2.IMWRITE_JPEG_QUALITY),
        quality
    ]

    success, encoded = cv2.imencode(
        ".jpg",
        image,
        encode_param
    )

    if not success:
        return image

    return cv2.imdecode(
        encoded,
        cv2.IMREAD_COLOR
    )


def add_noise(image):
    """
    Add very small sensor/compression noise.
    """

    noise = np.random.normal(
        0,
        3,
        image.shape
    ).astype(np.float32)

    result = image.astype(np.float32) + noise

    return np.clip(
        result,
        0,
        255
    ).astype(np.uint8)


def adjust_brightness_contrast(image):
    """
    Simulate different phone display/screenshot conditions.
    """

    alpha = random.uniform(
        0.90,
        1.10
    )

    beta = random.randint(
        -12,
        12
    )

    result = cv2.convertScaleAbs(
        image,
        alpha=alpha,
        beta=beta
    )

    return result


def slight_blur(image):
    """
    Simulate screen capture and resampling blur.
    """

    if random.random() < 0.5:
        return cv2.GaussianBlur(
            image,
            (3, 3),
            0
        )

    return image


def slight_resize(image):
    """
    Resize through a slightly different resolution
    to simulate mobile processing.
    """

    h, w = image.shape[:2]

    scale = random.uniform(
        0.85,
        1.15
    )

    new_w = max(
        32,
        int(w * scale)
    )

    new_h = max(
        32,
        int(h * scale)
    )

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    resized = cv2.resize(
        resized,
        (224, 224),
        interpolation=cv2.INTER_AREA
    )

    return resized


def create_mobile_variant(image):
    """
    Apply a combination of transformations.
    """

    result = image.copy()

    # Random horizontal flip
    if random.random() < 0.20:
        result = cv2.flip(
            result,
            1
        )

    # Brightness / contrast
    if random.random() < 0.80:
        result = adjust_brightness_contrast(
            result
        )

    # JPEG compression
    quality = random.randint(
        65,
        95
    )

    result = jpeg_compression(
        result,
        quality
    )

    # Noise
    if random.random() < 0.40:
        result = add_noise(
            result
        )

    # Blur
    result = slight_blur(
        result
    )

    # Resize
    result = slight_resize(
        result
    )

    return result


def main():

    print("=" * 70)
    print("TRUTHLENS - MOBILE ADAPTED DATASET CREATION")
    print("=" * 70)

    print()
    print("Source:")
    print(SOURCE_DIR)

    print()
    print("Output:")
    print(OUTPUT_DIR)

    print()
    print("Target per class:", IMAGES_PER_CLASS)

    # Create output directory
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    total_created = 0

    for class_name in CLASSES:

        print()
        print("-" * 70)
        print("CLASS:", class_name)
        print("-" * 70)

        source_class = os.path.join(
            SOURCE_DIR,
            class_name
        )

        output_class = os.path.join(
            OUTPUT_DIR,
            class_name
        )

        os.makedirs(
            output_class,
            exist_ok=True
        )

        if not os.path.exists(source_class):

            print(
                "ERROR: Source folder not found:",
                source_class
            )

            continue

        files = [
            f
            for f in os.listdir(source_class)
            if f.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".webp",
                    ".bmp"
                )
            )
        ]

        print(
            "Available source images:",
            len(files)
        )

        if len(files) == 0:
            print("No images found.")
            continue

        # Randomize source selection
        random.shuffle(files)

        created = 0

        while created < IMAGES_PER_CLASS:

            filename = random.choice(files)

            source_path = os.path.join(
                source_class,
                filename
            )

            try:

                image = cv2.imread(
                    source_path
                )

                if image is None:
                    continue

                image = cv2.resize(
                    image,
                    (224, 224),
                    interpolation=cv2.INTER_AREA
                )

                adapted = create_mobile_variant(
                    image
                )

                output_filename = (
                    f"{class_name.lower()}_"
                    f"mobile_{created + 1:04d}.jpg"
                )

                output_path = os.path.join(
                    output_class,
                    output_filename
                )

                cv2.imwrite(
                    output_path,
                    adapted,
                    [
                        int(cv2.IMWRITE_JPEG_QUALITY),
                        90
                    ]
                )

                created += 1

                if created % 50 == 0:

                    print(
                        f"Created: "
                        f"{created}/{IMAGES_PER_CLASS}"
                    )

            except Exception as e:

                print(
                    "Error processing:",
                    filename,
                    "|",
                    str(e)
                )

        print(
            f"{class_name}: "
            f"{created} images created"
        )

        total_created += created

    print()
    print("=" * 70)
    print("MOBILE ADAPTED DATASET COMPLETE")
    print("=" * 70)

    for class_name in CLASSES:

        folder = os.path.join(
            OUTPUT_DIR,
            class_name
        )

        if os.path.exists(folder):

            count = len([
                f
                for f in os.listdir(folder)
                if f.lower().endswith(
                    (
                        ".jpg",
                        ".jpeg",
                        ".png"
                    )
                )
            ])

            print(
                f"{class_name:<15}: {count:4d} images"
            )

    print("-" * 70)
    print(
        f"{'TOTAL':<15}: "
        f"{total_created:4d} images"
    )

    print()
    print("Output location:")
    print(OUTPUT_DIR)

    print()
    print("Expected structure:")

    for class_name in CLASSES:
        print(
            os.path.join(
                OUTPUT_DIR,
                class_name
            )
        )

    print()
    print("=" * 70)
    print("NEXT STEP: INSPECT THE MOBILE-ADAPTED DATASET")
    print("=" * 70)


if __name__ == "__main__":
    main()