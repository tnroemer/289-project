import os
import pandas as pd

from PIL import Image


DATA_ROOT = "/ocean/projects/mth250011p/troemer"
RUN_DIR = os.path.join(DATA_ROOT, "skin-lesions")

HAM_METADATA_PATH = os.path.join(RUN_DIR, "data", "ham10000-images", "metadata.csv")
MASK_DIR = os.path.join(RUN_DIR, "data", "predicted-masks")
LESION_IMAGE_DIR = os.path.join(RUN_DIR, "data", "lesion-white-images")
OUTPUT_METADATA_PATH = os.path.join(LESION_IMAGE_DIR, "metadata.csv")


def main():
    if not os.path.exists(HAM_METADATA_PATH):
        raise FileNotFoundError(
            f"Missing prepared HAM10000 metadata: {HAM_METADATA_PATH}. "
            "Run `sbatch submit/submit_create_data.sh` first."
        )

    df = pd.read_csv(HAM_METADATA_PATH)

    os.makedirs(LESION_IMAGE_DIR, exist_ok=True)

    rows = []
    created = 0
    skipped_existing = 0
    missing_images = 0
    missing_masks = 0

    for _, row in df.iterrows():
        image_id = row["image_id"]
        image_path = row["full_image_path"]
        mask_path = os.path.join(MASK_DIR, image_id + "_mask.png")
        output_path = os.path.join(LESION_IMAGE_DIR, image_id + ".jpg")

        if os.path.exists(output_path):
            skipped_existing += 1
        elif not os.path.exists(image_path):
            missing_images += 1
            continue
        elif not os.path.exists(mask_path):
            missing_masks += 1
            continue
        else:
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

        output_row = row.to_dict()
        output_row["original_image_path"] = image_path
        output_row["lesion_white_image_path"] = output_path
        output_row["lesion_image_path"] = output_path
        output_row["image_path"] = output_path
        rows.append(output_row)

    manifest_df = pd.DataFrame(rows)
    manifest_df.to_csv(OUTPUT_METADATA_PATH, index=False)

    print(f"Created lesion-white images: {created}")
    print(f"Already existed: {skipped_existing}")
    print(f"Missing original images: {missing_images}")
    print(f"Missing masks: {missing_masks}")
    print(f"Output folder: {LESION_IMAGE_DIR}")
    print(f"Saved metadata: {OUTPUT_METADATA_PATH}")


if __name__ == "__main__":
    main()
