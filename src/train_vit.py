import os

import pandas as pd
import torch
import wandb

from datasets import Dataset, Image, ClassLabel
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms


# -----------------------
# Global config
# -----------------------

os.environ["KAGGLEHUB_CACHE"] = "/ocean/projects/mth250011p/troemer/"

PROJECT_DIR = "/ocean/projects/mth250011p/troemer"
DATASET_DIR = os.path.join(PROJECT_DIR, "datasets", "skin-cancer-mnist-ham10000")
IMAGE_DIR = os.path.join(DATASET_DIR, "HAM10000_images")
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints")

image_size = 64
batch_size = 32
learning_rate = 1e-4
num_epochs = 50
num_workers = 4


# -----------------------
# Transforms
# -----------------------

# have to use mean and sd for our particular dataset in normalize

train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

val_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def train_transforms(examples):
    examples["pixel_values"] = [
        train_transform(image.convert("RGB"))
        for image in examples["image"]
    ]
    return examples


def val_transforms(examples):
    examples["pixel_values"] = [
        val_transform(image.convert("RGB"))
        for image in examples["image"]
    ]
    return examples


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["label"] for x in batch], dtype=torch.long),
    }


# -----------------------
# Model
# -----------------------

class BasicVIT(nn.Module):

    def __init__(

        self,

        num_classes,

        image_size=64,

        patch_size=8,

        embed_dim=128,

        depth=4,

        num_heads=4,

        mlp_dim=256,

        dropout=0.1,

    ):

        super().__init__()

        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_patches = (image_size // patch_size) ** 2

        # Split image into patches and turn each patch into an embedding vector

        self.patch_embed = nn.Conv2d(

            in_channels=3,

            out_channels=embed_dim,

            kernel_size=patch_size,

            stride=patch_size,

        )

        # Extra token that learns to summarize the whole image

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embeddings tell the transformer where each patch came from

        self.pos_embed = nn.Parameter(

            torch.zeros(1, self.num_patches + 1, embed_dim)

        )

        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(

            d_model=embed_dim,

            nhead=num_heads,

            dim_feedforward=mlp_dim,

            dropout=dropout,

            activation="gelu",

            batch_first=True,

            norm_first=True,

        )

        self.transformer = nn.TransformerEncoder(

            encoder_layer,

            num_layers=depth,

        )

        self.classifier = nn.Sequential(

            nn.LayerNorm(embed_dim),

            nn.Linear(embed_dim, num_classes),

        )

        self._init_weights()

    def _init_weights(self):

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):

        batch_size = x.shape[0]

        x = self.patch_embed(x)

        # [B, embed_dim, H/patch_size, W/patch_size]

        x = x.flatten(2)

        # [B, embed_dim, num_patches]

        x = x.transpose(1, 2)

        # [B, num_patches, embed_dim]

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)

        # [B, num_patches + 1, embed_dim]

        x = x + self.pos_embed

        x = self.dropout(x)

        x = self.transformer(x)

        x = x[:, 0]

        # Use class token for classification

        x = self.classifier(x)

        return x


# -----------------------
# Data setup
# -----------------------

def prepare_dataset():
    csv_path = "/ocean/projects/mth250011p/troemer/datasets/kmader/skin-cancer-mnist-ham10000/versions/2/HAM10000_metadata.csv"

    df = pd.read_csv(csv_path)
    df["image"] = df["image_id"].apply(
        lambda x: os.path.join(IMAGE_DIR, x + ".jpg")
    )

    labels = sorted(df["dx"].unique())
    label_feature = ClassLabel(names=labels)

    df["label"] = df["dx"].apply(lambda x: label_feature.str2int(x))

    dataset = Dataset.from_pandas(df[["image", "label"]])
    dataset = dataset.cast_column("image", Image())
    dataset = dataset.cast_column("label", label_feature)

    splits = dataset.train_test_split(
        test_size=0.2,
        stratify_by_column="label",
        seed=42,
    )

    train_ds = splits["train"].with_transform(train_transforms)
    val_ds = splits["test"].with_transform(val_transforms)

    return train_ds, val_ds, labels


# -----------------------
# Training / evaluation
# -----------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        images = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


# -----------------------
# Main
# -----------------------

def main():
    train_ds, val_ds, labels = prepare_dataset()

    num_classes = len(labels)

    config = {
        "model": "ViT",
        "dataset": "HAM10000",
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "image_size": image_size,
        "num_classes": num_classes,
        "num_workers": num_workers,
        "patch_size": 8,
        "embed_dim": 128,
        "depth": 4,
        "num_heads": 4,
        "mlp_dim": 256,
        "dropout": 0.1,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    wandb.login(key=os.environ["WANDB_API_KEY"])

    run = wandb.init(
        entity="tnroemer-berk",
        project="skin-cancer-cnn",
        name="vit",
        config=config,
    )

    model = BasicVIT(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), 
                      lr=learning_rate,
                      weight_decay=0.05)

    # Optional. This can add overhead, but it is useful for debugging.
    # wandb.watch(model, log="gradients", log_freq=100)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model_vit.pt")

    best_val_acc = 0.0
    patience = 8
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
        })

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "labels": labels,
                "config": config,
            }

            torch.save(checkpoint, best_model_path)

            wandb.run.summary["best_val_accuracy"] = best_val_acc
            wandb.run.summary["best_epoch"] = epoch + 1

            wandb.save(best_model_path)

            print(f"Saved new best model: val_acc={val_acc:.4f}")
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    wandb.finish()


if __name__ == "__main__":
    main()