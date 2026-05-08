import os
import shutil

import kagglehub
import pandas as pd


os.environ["KAGGLEHUB_CACHE"] = "/ocean/projects/mth250011p/troemer/"

DATA_ROOT = "/ocean/projects/mth250011p/troemer"
RUN_DIR = os.path.join(DATA_ROOT, "skin-lesions")
DATASET_DIR = os.path.join(DATA_ROOT, "datasets", "skin-cancer-mnist-ham10000")
KAGGLE_DATASET_DIR = os.path.join(
    DATA_ROOT,
    "datasets",
    "kmader",
    "skin-cancer-mnist-ham10000",
    "versions",
    "2",
)
IMAGE_DIR = os.path.join(DATASET_DIR, "HAM10000_images")
LESION_IMAGE_DIR = os.path.join(RUN_DIR, "data", "lesion-white-images")
SPLIT_DIR = os.path.join(RUN_DIR, "data", "splits")

COMMON_LABELS = ["akiec", "bcc", "bkl", "mel", "nv"]
HAM_TO_PAD_LABELS = {
    "akiec": "ACK",
    "bcc": "BCC",
    "bkl": "SEK",
    "mel": "MEL",
    "nv": "NEV",
}

seed = 42


def find_metadata_csv():
    local_csv = os.path.join(DATASET_DIR, "HAM10000_metadata.csv")
    kaggle_csv = os.path.join(KAGGLE_DATASET_DIR, "HAM10000_metadata.csv")

    if os.path.exists(local_csv):
        return local_csv

    if os.path.exists(kaggle_csv):
        return kaggle_csv

    path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")

    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)

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
    shutil.copy2(
        os.path.join(path, "HAM10000_metadata.csv"),
        local_csv,
    )

    return local_csv


def find_image_path(image_id):
    local_path = os.path.join(IMAGE_DIR, image_id + ".jpg")
    kaggle_path_1 = os.path.join(KAGGLE_DATASET_DIR, "HAM10000_images_part_1", image_id + ".jpg")
    kaggle_path_2 = os.path.join(KAGGLE_DATASET_DIR, "HAM10000_images_part_2", image_id + ".jpg")

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


def main():
    metadata_path = find_metadata_csv()
    df = pd.read_csv(metadata_path)

    df["dx"] = df["dx"].str.lower()
    df = df[df["dx"].isin(COMMON_LABELS)].copy().reset_index(drop=True)

    label_to_id = {label: i for i, label in enumerate(COMMON_LABELS)}

    df["label"] = df["dx"].map(label_to_id)
    df["full_image_path"] = df["image_id"].apply(find_image_path)
    df["lesion_image_path"] = df["image_id"].apply(
        lambda x: os.path.join(LESION_IMAGE_DIR, x + ".jpg")
    )
    df["pad_label"] = df["dx"].map(HAM_TO_PAD_LABELS)

    missing_full_images = (~df["full_image_path"].apply(os.path.exists)).sum()

    keep_columns = [
        "image_id",
        "lesion_id",
        "dx",
        "label",
        "pad_label",
        "full_image_path",
        "lesion_image_path",
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

    os.makedirs(SPLIT_DIR, exist_ok=True)

    train_path = os.path.join(SPLIT_DIR, "ham10000_train.csv")
    val_path = os.path.join(SPLIT_DIR, "ham10000_val.csv")
    test_path = os.path.join(SPLIT_DIR, "ham10000_test.csv")
    all_path = os.path.join(SPLIT_DIR, "ham10000_all.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    pd.concat([train_df, val_df, test_df]).to_csv(all_path, index=False)

    print(f"Metadata path: {metadata_path}")
    print(f"Common HAM/PAD labels: {COMMON_LABELS}")
    print("PAD mapping:", HAM_TO_PAD_LABELS)
    print(f"Missing full images: {missing_full_images}")
    print("Train counts:")
    print(train_df["dx"].value_counts().sort_index())
    print("Val counts:")
    print(val_df["dx"].value_counts().sort_index())
    print("Test counts:")
    print(test_df["dx"].value_counts().sort_index())
    print(f"Saved train split to {train_path}")
    print(f"Saved val split to {val_path}")
    print(f"Saved test split to {test_path}")


if __name__ == "__main__":
    main()
