import os
import shutil
import zipfile

import kagglehub
import pandas as pd

from PIL import Image


os.environ["KAGGLEHUB_CACHE"] = "/ocean/projects/mth250011p/troemer/"

DATA_ROOT = "/ocean/projects/mth250011p/troemer"
RUN_DIR = os.path.join(DATA_ROOT, "skin-lesions")

HAM_DATASET_DIR = os.path.join(DATA_ROOT, "datasets", "skin-cancer-mnist-ham10000")
HAM_KAGGLE_DATASET_DIR = os.path.join(
    DATA_ROOT,
    "datasets",
    "kmader",
    "skin-cancer-mnist-ham10000",
    "versions",
    "2",
)
HAM_SOURCE_IMAGE_DIR = os.path.join(HAM_DATASET_DIR, "HAM10000_images")
HAM_IMAGE_DIR = os.path.join(RUN_DIR, "data", "ham10000-images")
HAM_METADATA_PATH = os.path.join(HAM_IMAGE_DIR, "metadata.csv")
HAM_LESION_IMAGE_DIR = os.path.join(RUN_DIR, "data", "lesion-white-images")
HAM_SPLIT_DIR = os.path.join(RUN_DIR, "data", "splits")

PAD_RAW_DATASET_DIR = os.path.join(DATA_ROOT, "datasets", "pad-ufes-20")
PAD_IMAGE_DIR = os.path.join(RUN_DIR, "data", "pad-ufes-20-images")
PAD_METADATA_PATH = os.path.join(PAD_IMAGE_DIR, "metadata.csv")

COMMON_LABELS = ["akiec", "bcc", "bkl", "mel", "nv"]
HAM_TO_PAD_LABELS = {
    "akiec": "ACK",
    "bcc": "BCC",
    "bkl": "SEK",
    "mel": "MEL",
    "nv": "NEV",
}
PAD_TO_HAM_LABELS = {
    "ACK": "akiec",
    "BCC": "bcc",
    "SEK": "bkl",
    "MEL": "mel",
    "NEV": "nv",
}

seed = 42


def find_ham_metadata_csv():
    local_csv = os.path.join(HAM_DATASET_DIR, "HAM10000_metadata.csv")
    kaggle_csv = os.path.join(HAM_KAGGLE_DATASET_DIR, "HAM10000_metadata.csv")

    if os.path.exists(local_csv):
        return local_csv

    if os.path.exists(kaggle_csv):
        return kaggle_csv

    path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")

    os.makedirs(HAM_DATASET_DIR, exist_ok=True)
    os.makedirs(HAM_SOURCE_IMAGE_DIR, exist_ok=True)

    shutil.copytree(
        os.path.join(path, "HAM10000_images_part_1"),
        HAM_SOURCE_IMAGE_DIR,
        dirs_exist_ok=True,
    )
    shutil.copytree(
        os.path.join(path, "HAM10000_images_part_2"),
        HAM_SOURCE_IMAGE_DIR,
        dirs_exist_ok=True,
    )
    shutil.copy2(
        os.path.join(path, "HAM10000_metadata.csv"),
        local_csv,
    )

    return local_csv


def find_ham_source_image_path(image_id):
    local_path = os.path.join(HAM_SOURCE_IMAGE_DIR, image_id + ".jpg")
    kaggle_path_1 = os.path.join(HAM_KAGGLE_DATASET_DIR, "HAM10000_images_part_1", image_id + ".jpg")
    kaggle_path_2 = os.path.join(HAM_KAGGLE_DATASET_DIR, "HAM10000_images_part_2", image_id + ".jpg")

    if os.path.exists(local_path):
        return local_path
    if os.path.exists(kaggle_path_1):
        return kaggle_path_1
    if os.path.exists(kaggle_path_2):
        return kaggle_path_2

    return local_path


def split_group(group):
    group = group.sample(frac=1, random_state=seed).reset_index(drop=True)

    n = len(group)
    n_test = max(1, round(n * 0.15))
    n_val = max(1, round(n * 0.15))

    test_df = group.iloc[:n_test]
    val_df = group.iloc[n_test:n_test + n_val]
    train_df = group.iloc[n_test + n_val:]

    return train_df, val_df, test_df


def create_ham10000_data():
    metadata_path = find_ham_metadata_csv()
    df = pd.read_csv(metadata_path)

    df["dx"] = df["dx"].str.lower()
    df = df[df["dx"].isin(COMMON_LABELS)].copy().reset_index(drop=True)

    label_to_id = {label: i for i, label in enumerate(COMMON_LABELS)}

    df["label"] = df["dx"].map(label_to_id)
    df["pad_label"] = df["dx"].map(HAM_TO_PAD_LABELS)

    os.makedirs(HAM_IMAGE_DIR, exist_ok=True)

    rows = []
    copied = 0
    skipped_existing = 0
    missing_full_images = 0

    for _, row in df.iterrows():
        image_id = row["image_id"]
        source_path = find_ham_source_image_path(image_id)
        output_path = os.path.join(HAM_IMAGE_DIR, image_id + ".jpg")

        if not os.path.exists(source_path):
            missing_full_images += 1
            continue

        if os.path.exists(output_path):
            skipped_existing += 1
        else:
            with Image.open(source_path) as image:
                image = image.convert("RGB")
                image.save(output_path, quality=95)
            copied += 1

        output_row = row.to_dict()
        output_row["original_image_path"] = source_path
        output_row["full_image_path"] = output_path
        output_row["image_path"] = output_path
        output_row["lesion_image_path"] = os.path.join(HAM_LESION_IMAGE_DIR, image_id + ".jpg")
        rows.append(output_row)

    df = pd.DataFrame(rows)

    keep_columns = [
        "image_id",
        "lesion_id",
        "dx",
        "label",
        "pad_label",
        "original_image_path",
        "full_image_path",
        "lesion_image_path",
        "image_path",
    ]
    df = df[keep_columns]

    train_parts = []
    val_parts = []
    test_parts = []

    for _, group in df.groupby("label"):
        train_df, val_df, test_df = split_group(group)
        train_parts.append(train_df)
        val_parts.append(val_df)
        test_parts.append(test_df)

    train_df = pd.concat(train_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_parts).sample(frac=1, random_state=seed).reset_index(drop=True)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    os.makedirs(HAM_SPLIT_DIR, exist_ok=True)

    train_path = os.path.join(HAM_SPLIT_DIR, "ham10000_train.csv")
    val_path = os.path.join(HAM_SPLIT_DIR, "ham10000_val.csv")
    test_path = os.path.join(HAM_SPLIT_DIR, "ham10000_test.csv")
    all_path = os.path.join(HAM_SPLIT_DIR, "ham10000_all.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    all_df = pd.concat([train_df, val_df, test_df])
    all_df.to_csv(all_path, index=False)
    all_df.to_csv(HAM_METADATA_PATH, index=False)

    print("Created HAM10000 common-label image folder and splits")
    print(f"Metadata path: {metadata_path}")
    print(f"Copied HAM images: {copied}")
    print(f"Already existed: {skipped_existing}")
    print(f"Missing HAM full images: {missing_full_images}")
    print("Train counts:")
    print(train_df["dx"].value_counts().sort_index())
    print("Val counts:")
    print(val_df["dx"].value_counts().sort_index())
    print("Test counts:")
    print(test_df["dx"].value_counts().sort_index())
    print(f"Saved train split to {train_path}")
    print(f"Saved val split to {val_path}")
    print(f"Saved test split to {test_path}")
    print(f"Output folder: {HAM_IMAGE_DIR}")
    print(f"Saved HAM metadata: {HAM_METADATA_PATH}")


def download_pad_dataset():
    if os.path.exists(PAD_RAW_DATASET_DIR):
        extract_zip_files(PAD_RAW_DATASET_DIR)
        return PAD_RAW_DATASET_DIR

    path = kagglehub.dataset_download("mahdavi1202/skin-cancer")
    os.makedirs(PAD_RAW_DATASET_DIR, exist_ok=True)

    for item in os.listdir(path):
        source_path = os.path.join(path, item)
        target_path = os.path.join(PAD_RAW_DATASET_DIR, item)

        if os.path.isdir(source_path):
            shutil.copytree(source_path, target_path, dirs_exist_ok=True)
        elif not os.path.exists(target_path):
            shutil.copy2(source_path, target_path)

    extract_zip_files(PAD_RAW_DATASET_DIR)

    return PAD_RAW_DATASET_DIR


def extract_zip_files(dataset_dir):
    for root, _, files in os.walk(dataset_dir):
        for filename in files:
            if not filename.lower().endswith(".zip"):
                continue

            zip_path = os.path.join(root, filename)
            extract_dir = os.path.join(root, os.path.splitext(filename)[0])

            if os.path.exists(extract_dir):
                continue

            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zip_file:
                zip_file.extractall(extract_dir)


def find_pad_metadata_csv(dataset_dir):
    for root, _, files in os.walk(dataset_dir):
        for filename in files:
            if not filename.lower().endswith(".csv"):
                continue

            csv_path = os.path.join(root, filename)
            try:
                df = pd.read_csv(csv_path, nrows=5)
            except Exception:
                continue

            columns = set(df.columns)
            if {"img_id", "diagnostic"}.issubset(columns):
                return csv_path

    raise FileNotFoundError("Could not find PAD-UFES-20 metadata CSV with img_id and diagnostic columns.")


def make_pad_image_index(dataset_dir):
    image_index = {}
    image_extensions = {".png", ".jpg", ".jpeg"}

    for root, _, files in os.walk(dataset_dir):
        for filename in files:
            stem, ext = os.path.splitext(filename)
            if ext.lower() not in image_extensions:
                continue

            path = os.path.join(root, filename)
            image_index[filename] = path
            image_index[stem] = path

    return image_index


def create_pad_ufes20_data():
    dataset_dir = download_pad_dataset()
    metadata_path = find_pad_metadata_csv(dataset_dir)
    image_index = make_pad_image_index(dataset_dir)
    label_to_id = {label: i for i, label in enumerate(COMMON_LABELS)}

    df = pd.read_csv(metadata_path)
    df["diagnostic"] = df["diagnostic"].astype(str).str.upper()
    df["common_label"] = df["diagnostic"].map(PAD_TO_HAM_LABELS)
    df = df.dropna(subset=["common_label"]).copy().reset_index(drop=True)
    df["label"] = df["common_label"].map(label_to_id)

    os.makedirs(PAD_IMAGE_DIR, exist_ok=True)

    rows = []
    copied = 0
    skipped_existing = 0
    missing_images = 0

    for _, row in df.iterrows():
        img_id = str(row["img_id"])
        img_key = os.path.splitext(img_id)[0]
        source_path = image_index.get(img_id) or image_index.get(img_key) or image_index.get(img_key + ".png")

        if source_path is None:
            missing_images += 1
            continue

        output_path = os.path.join(PAD_IMAGE_DIR, img_key + ".jpg")

        if os.path.exists(output_path):
            skipped_existing += 1
        else:
            with Image.open(source_path) as image:
                image = image.convert("RGB")
                image.save(output_path, quality=95)
            copied += 1

        output_row = row.to_dict()
        output_row["image_id"] = img_key
        output_row["original_image_path"] = source_path
        output_row["image_path"] = output_path
        rows.append(output_row)

    manifest_df = pd.DataFrame(rows)
    manifest_df["dx"] = manifest_df["common_label"].str.lower()
    manifest_df.to_csv(PAD_METADATA_PATH, index=False)

    train_parts = []
    val_parts = []
    test_parts = []

    for _, group in manifest_df.groupby("label"):
        train_df, val_df, test_df = split_group(group)
        train_parts.append(train_df)
        val_parts.append(val_df)
        test_parts.append(test_df)

    train_df = pd.concat(train_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_parts).sample(frac=1, random_state=seed).reset_index(drop=True)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    os.makedirs(HAM_SPLIT_DIR, exist_ok=True)

    train_path = os.path.join(HAM_SPLIT_DIR, "pad_ufes20_train.csv")
    val_path = os.path.join(HAM_SPLIT_DIR, "pad_ufes20_val.csv")
    test_path = os.path.join(HAM_SPLIT_DIR, "pad_ufes20_test.csv")
    all_path = os.path.join(HAM_SPLIT_DIR, "pad_ufes20_all.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    pd.concat([train_df, val_df, test_df]).to_csv(all_path, index=False)

    print("Created PAD-UFES-20 common-label image folder and splits")
    print(f"Metadata path: {metadata_path}")
    print(f"Common HAM/PAD labels: {COMMON_LABELS}")
    print("PAD mapping:", PAD_TO_HAM_LABELS)
    print(f"Copied PAD images: {copied}")
    print(f"Already existed: {skipped_existing}")
    print(f"Missing PAD source images: {missing_images}")
    print("PAD counts:")
    print(manifest_df["common_label"].value_counts().sort_index())
    print("Train counts:")
    print(train_df["common_label"].value_counts().sort_index())
    print("Val counts:")
    print(val_df["common_label"].value_counts().sort_index())
    print("Test counts:")
    print(test_df["common_label"].value_counts().sort_index())
    print(f"Output folder: {PAD_IMAGE_DIR}")
    print(f"Saved PAD metadata: {PAD_METADATA_PATH}")
    print(f"Saved train split to {train_path}")
    print(f"Saved val split to {val_path}")
    print(f"Saved test split to {test_path}")


def main():
    create_ham10000_data()
    create_pad_ufes20_data()


if __name__ == "__main__":
    main()
