import os

import pandas as pd
import torch
import wandb

from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models.model_architectures import build_segmentation_model


DATA_ROOT = "/ocean/projects/mth250011p/troemer"
RUN_DIR = os.path.join(DATA_ROOT, "skin-lesions")
SPLIT_DIR = os.path.join(RUN_DIR, "data", "splits")
HAM_METADATA_PATH = os.path.join(RUN_DIR, "data", "ham10000-images", "metadata.csv")
MASK_DIR = os.path.join(RUN_DIR, "data", "predicted-masks")
MODEL_DIR = os.path.join(RUN_DIR, "models")
METRICS_DIR = os.path.join(RUN_DIR, "metrics")

model_path = os.path.join(MODEL_DIR, "segmentation_deeplabv3_resnet50.pth")
metrics_path = os.path.join(METRICS_DIR, "segmentation_deeplabv3_resnet50_metrics.csv")

image_size = 320
batch_size = 8
num_epochs = 40
num_workers = 4
learning_rate = 1e-4
mask_threshold = 0.5
seed = 42


class HamSegmentationDataset(Dataset):
    def __init__(self, df, augment=False):
        self.df = df.reset_index(drop=True)
        self.augment = augment
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        self.color_jitter = transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02,
        )
        self.mask_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        with Image.open(row["full_image_path"]) as image, Image.open(row["mask_path"]) as mask:
            image = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
            mask = mask.convert("L").resize((image_size, image_size), Image.NEAREST)

            if self.augment:
                if torch.rand(1).item() < 0.5:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
                if torch.rand(1).item() < 0.5:
                    image = image.transpose(Image.FLIP_TOP_BOTTOM)
                    mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

                rotations = int(torch.randint(0, 4, (1,)).item())
                if rotations:
                    angle = 90 * rotations
                    image = image.rotate(angle, resample=Image.BILINEAR)
                    mask = mask.rotate(angle, resample=Image.NEAREST)
                if torch.rand(1).item() < 0.5:
                    image = self.color_jitter(image)

            image = self.image_transform(image)
            mask = (self.mask_transform(mask) > 0.5).float()

        return {
            "pixel_values": image,
            "mask": mask,
            "image_id": row["image_id"],
        }


def add_mask_paths(df):
    df = df.copy()
    df["mask_path"] = df["image_id"].apply(lambda image_id: os.path.join(MASK_DIR, image_id + "_mask.png"))
    df = df[df["full_image_path"].apply(os.path.exists)].reset_index(drop=True)
    df = df[df["mask_path"].apply(os.path.exists)].reset_index(drop=True)
    return df


def load_split(split_name):
    split_path = os.path.join(SPLIT_DIR, f"ham10000_{split_name}.csv")
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"Missing HAM10000 split file: {split_path}. "
            "Run `sbatch submit/submit_create_data.sh` first."
        )

    df = add_mask_paths(pd.read_csv(split_path))
    if len(df) == 0:
        raise FileNotFoundError(
            f"No HAM10000 images with masks found for split {split_name}. "
            f"Expected masks like {os.path.join(MASK_DIR, 'ISIC_0024306_mask.png')}."
        )

    print(f"{split_name} segmentation examples: {len(df)}")
    return df


def dice_loss(logits, masks):
    probs = torch.sigmoid(logits)
    intersection = (probs * masks).sum(dim=(1, 2, 3))
    denominator = probs.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
    dice = (2 * intersection + 1e-6) / (denominator + 1e-6)
    return 1 - dice.mean()


def segmentation_metrics(logits, masks):
    preds = (torch.sigmoid(logits) >= mask_threshold).float()
    intersection = (preds * masks).sum(dim=(1, 2, 3))
    pred_area = preds.sum(dim=(1, 2, 3))
    mask_area = masks.sum(dim=(1, 2, 3))
    union = pred_area + mask_area - intersection

    dice = ((2 * intersection + 1e-6) / (pred_area + mask_area + 1e-6)).mean().item()
    iou = ((intersection + 1e-6) / (union + 1e-6)).mean().item()
    return dice, iou


def model_logits(output):
    if isinstance(output, dict):
        return output["out"]
    return output


def run_epoch(model, loader, optimizer, criterion, device, train):
    model.train(train)

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total = 0

    for batch in loader:
        images = batch["pixel_values"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            output = model(images)
            logits = model_logits(output)
            loss = criterion(logits, masks) + dice_loss(logits, masks)
            if train and isinstance(output, dict) and "aux" in output:
                aux_loss = criterion(output["aux"], masks) + dice_loss(output["aux"], masks)
                loss = loss + 0.4 * aux_loss

            if train:
                loss.backward()
                optimizer.step()

        dice, iou = segmentation_metrics(logits.detach(), masks)
        batch_size_actual = images.size(0)
        total_loss += loss.item() * batch_size_actual
        total_dice += dice * batch_size_actual
        total_iou += iou * batch_size_actual
        total += batch_size_actual

    return {
        "loss": total_loss / total,
        "dice": total_dice / total,
        "iou": total_iou / total,
    }


def save_ham_masks(model, device):
    if not os.path.exists(HAM_METADATA_PATH):
        raise FileNotFoundError(f"Missing HAM10000 metadata: {HAM_METADATA_PATH}")

    df = pd.read_csv(HAM_METADATA_PATH)
    os.makedirs(MASK_DIR, exist_ok=True)

    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    model.eval()
    saved = 0
    missing_images = 0

    with torch.no_grad():
        for _, row in df.iterrows():
            image_id = row["image_id"]
            image_path = row["full_image_path"]
            output_path = os.path.join(MASK_DIR, image_id + "_mask.png")

            if not os.path.exists(image_path):
                missing_images += 1
                continue

            with Image.open(image_path) as image:
                image = image.convert("RGB")
                original_size = image.size
                tensor = image_transform(image).unsqueeze(0).to(device)
                probs = torch.sigmoid(model_logits(model(tensor)))

            mask_array = (probs.squeeze().cpu().numpy() >= mask_threshold).astype("uint8") * 255
            mask = Image.fromarray(mask_array).convert("L")
            mask = mask.resize(original_size, Image.NEAREST)
            mask.save(output_path)
            saved += 1

    print(f"Saved HAM predicted masks: {saved}")
    print(f"Missing HAM images while saving masks: {missing_images}")
    print(f"Mask folder: {MASK_DIR}")


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main():
    torch.manual_seed(seed)

    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    train_ds = HamSegmentationDataset(train_df, augment=True)
    val_ds = HamSegmentationDataset(val_df, augment=False)
    test_ds = HamSegmentationDataset(test_df, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    config = {
        "model": "DeepLabV3ResNet50",
        "out_channels": 1,
        "image_size": image_size,
        "mask_threshold": mask_threshold,
        "use_pretrained_backbone": True,
        "aux_loss": True,
        "dataset": "ham10000",
        "target": "lesion_segmentation_mask",
        "mask_source": MASK_DIR,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "num_workers": num_workers,
        "loss": "BCEWithLogitsLoss + DiceLoss",
        "scheduler": "cosine_annealing_lr",
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
    }
    model = build_segmentation_model(config).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    if "WANDB_API_KEY" in os.environ:
        wandb.login(key=os.environ["WANDB_API_KEY"])

    wandb.init(
        project="skin-cancer-cnn",
        name="ham10000-segmentation-deeplabv3-resnet50",
        config=config,
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    rows = []
    best_val_dice = -1.0
    patience = 10
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, criterion, device, train=False)
        scheduler.step()

        rows.append({"epoch": epoch, "split": "train", **train_metrics})
        rows.append({"epoch": epoch, "split": "val", **val_metrics})

        wandb.log({
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train/loss": train_metrics["loss"],
            "train/dice": train_metrics["dice"],
            "train/iou": train_metrics["iou"],
            "val/loss": val_metrics["loss"],
            "val/dice": val_metrics["dice"],
            "val/iou": val_metrics["iou"],
        })

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Dice: {train_metrics['dice']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Dice: {val_metrics['dice']:.4f} | "
            f"Val IoU: {val_metrics['iou']:.4f}"
        )

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            epochs_without_improvement = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_dice": best_val_dice,
                "config": {
                    **config,
                    "use_pretrained_backbone": False,
                },
            }
            torch.save(checkpoint, model_path)
            wandb.run.summary["best_epoch"] = epoch
            wandb.run.summary["best_val_dice"] = val_metrics["dice"]
            wandb.run.summary["best_val_iou"] = val_metrics["iou"]
            wandb.run.summary["best_val_loss"] = val_metrics["loss"]
            wandb.save(model_path)
            print(f"Saved new best segmentation model: val_dice={best_val_dice:.4f}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    checkpoint = load_checkpoint(model_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = run_epoch(model, test_loader, optimizer, criterion, device, train=False)
    rows.append({"epoch": checkpoint["epoch"], "split": "test", **test_metrics})

    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    wandb.log({
        "test/loss": test_metrics["loss"],
        "test/dice": test_metrics["dice"],
        "test/iou": test_metrics["iou"],
    })
    wandb.run.summary["test_loss"] = test_metrics["loss"]
    wandb.run.summary["test_dice"] = test_metrics["dice"]
    wandb.run.summary["test_iou"] = test_metrics["iou"]
    wandb.save(metrics_path)

    print(f"Best validation dice: {checkpoint['best_val_dice']:.4f}")
    print(f"Test Dice: {test_metrics['dice']:.4f}")
    print(f"Test IoU: {test_metrics['iou']:.4f}")
    print(f"Saved segmentation model: {model_path}")
    print(f"Saved segmentation metrics: {metrics_path}")

    save_ham_masks(model, device)
    wandb.finish()


if __name__ == "__main__":
    main()
