import os
import shutil

import kagglehub
import pandas as pd

from PIL import Image


# -----------------------
# Global config
# -----------------------

os.environ["KAGGLEHUB_CACHE"] = "/ocean/projects/mth250011p/troemer/"

DATA_ROOT = "/ocean/projects/mth250011p/troemer"
RUN_DIR = os.path.join(DATA_ROOT, "skin-lesions")

DATASET_DIR = os.path.join(DATA_ROOT, "datasets", "skin-cancer-mnist-ham10000")
IMAGE_DIR = os.path.join(DATASET_DIR, "HAM10000_images")
MASK_DIR = os.path.join(RUN_DIR, "data", "predicted-masks")
LESION_IMAGE_DIR = os.path.join(RUN_DIR, "data", "lesion-white-images")


def main():
    if not os.path.exists(DATASET_DIR):
        path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")

        os.makedirs(DATASET_DIR, exist_ok=True)

        shutil.copytree(
            os.path.join(path, "HAM10000_images_part_1"),
            IMAGE_DIR,
            dirs_exist_ok=True,
        )

        shutil.copytree(
            os.path.join(path, "HAM10000_images_part_2"),
            IMAGE_DIR,
            dirs_exist_ok=True,
        )

        csv_path = os.path.join(path, "HAM10000_metadata.csv")
    else:
        csv_path = "/ocean/projects/mth250011p/troemer/datasets/kmader/skin-cancer-mnist-ham10000/versions/2/HAM10000_metadata.csv"

    df = pd.read_csv(csv_path)

    os.makedirs(LESION_IMAGE_DIR, exist_ok=True)

    created = 0
    skipped_existing = 0
    missing_images = 0
    missing_masks = 0

    for image_id in df["image_id"]:
        image_path = os.path.join(IMAGE_DIR, image_id + ".jpg")
        mask_path = os.path.join(MASK_DIR, image_id + "_mask.png")
        output_path = os.path.join(LESION_IMAGE_DIR, image_id + ".jpg")

        if os.path.exists(output_path):
            skipped_existing += 1
            continue

        if not os.path.exists(image_path):
            missing_images += 1
            continue

        if not os.path.exists(mask_path):
            missing_masks += 1
            continue

        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            image = image.convert("RGB")
            mask = mask.convert("L")

            if mask.size != image.size:
                mask = mask.resize(image.size, Image.NEAREST)

            mask = mask.point(lambda p: 255 if p > 127 else 0)

            white_background = Image.new("RGB", image.size, (255, 255, 255))
            lesion_image = Image.composite(image, white_background, mask)
            lesion_image.save(output_path, quality=95)

        created += 1

    print(f"Created lesion-white images: {created}")
    print(f"Already existed: {skipped_existing}")
    print(f"Missing original images: {missing_images}")
    print(f"Missing masks: {missing_masks}")
    print(f"Output folder: {LESION_IMAGE_DIR}")


if __name__ == "__main__":
    main()
