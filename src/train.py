import os
import kagglehub
import os
import pandas as pd
from datasets import Dataset, Image, ClassLabel
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
import wandb
import shutil

os.environ["KAGGLEHUB_CACHE"] = "/ocean/projects/mth250011p/troemer/"
work_path = '/ocean/projects/mth250011p/troemer/datasets/skin-cancer-mnist-ham10000'

if not os.path.exists("/ocean/projects/mth250011p/troemer/datasets/skin-cancer-mnist-ham10000"):
    # Download latest version
    path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
    metadata = pd.read_csv(path + "/HAM10000_metadata.csv")

    shutil.copytree(path + "/HAM10000_images_part_1", work_path + "/HAM10000_images", dirs_exist_ok=True)
    shutil.copytree(path + "/HAM10000_images_part_2", work_path + "/HAM10000_images", dirs_exist_ok=True)

image_dir = work_path + "/HAM10000_images"
csv_path = "/ocean/projects/mth250011p/troemer/datasets/kmader/skin-cancer-mnist-ham10000/versions/2/HAM10000_metadata.csv"

df = pd.read_csv(csv_path)
df["image"] = df["image_id"].apply(lambda x: os.path.join(image_dir, x + ".jpg"))

labels = sorted(df["dx"].unique())
label_feature = ClassLabel(names=labels)

df["label"] = df["dx"].apply(lambda x: label_feature.str2int(x))

dataset = Dataset.from_pandas(df[["image", "label"]])
dataset = dataset.cast_column("image", Image())
dataset = dataset.cast_column("label", label_feature)

splits = dataset.train_test_split(test_size=0.2, stratify_by_column="label", seed=42)
train_ds = splits["train"]
val_ds = splits["test"]

num_classes = len(labels)

image_size = 128
batch_size = 32
learning_rate = 1e-4
num_epochs = 10

config = {
    "model": "BasicCNN",
    "dataset": "HAM10000",
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "epochs": num_epochs,
    "image_size": image_size,
    "num_classes": num_classes,
}

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

def train_transforms(example):
    example["pixel_values"] = train_transform(example["image"].convert("RGB"))
    return example

def val_transforms(example):
    example["pixel_values"] = val_transform(example["image"].convert("RGB"))
    return example

train_ds = train_ds.with_transform(train_transforms)
val_ds = val_ds.with_transform(val_transforms)

def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["label"] for x in batch]),
    }

class BasicCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 128 -> 64

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 64 -> 32

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 32 -> 16

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 16 -> 8
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for batch in loader:
        images = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad()

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

    total_loss = 0
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
def main():

    # dataset setup

    # train_ds / val_ds setup

    # transforms

    # dataloaders

    train_loader = DataLoader(

        train_ds,

        batch_size=batch_size,

        shuffle=True,

        collate_fn=collate_fn,

        num_workers=4,

        pin_memory=True,

    )

    val_loader = DataLoader(

        val_ds,

        batch_size=batch_size,

        shuffle=False,

        collate_fn=collate_fn,

        num_workers=4,

        pin_memory=True,

    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.login(key=os.environ["WANDB_API_KEY"])

    wandb.init(

        entity="tnroemer-berk",

        project="skin-cancer-cnn",

        name="basic-cnn",

        config=config,

    )

    model = BasicCNN(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    wandb.watch(model, log="gradients", log_freq=100)

    best_val_acc = 0.0

    os.makedirs("checkpoints", exist_ok=True)

    best_model_path = "checkpoints/best_model.pt"

    for epoch in range(num_epochs):

        train_loss, train_acc = train_one_epoch(

            model, train_loader, optimizer, criterion, device

        )

        val_loss, val_acc = evaluate(

            model, val_loader, criterion, device

        )

        wandb.log({

            "epoch": epoch + 1,

            "train/loss": train_loss,

            "train/accuracy": train_acc,

            "val/loss": val_loss,

            "val/accuracy": val_acc,

        })

        print(

            f"Epoch {epoch+1}/{num_epochs} | "

            f"Train Loss: {train_loss:.4f} | "

            f"Train Acc: {train_acc:.4f} | "

            f"Val Loss: {val_loss:.4f} | "

            f"Val Acc: {val_acc:.4f}"

        )

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            torch.save({

                "epoch": epoch + 1,

                "model_state_dict": model.state_dict(),

                "optimizer_state_dict": optimizer.state_dict(),

                "val_acc": val_acc,

                "val_loss": val_loss,

                "labels": labels,

                "config": config,

            }, best_model_path)

            wandb.run.summary["best_val_accuracy"] = best_val_acc

            wandb.run.summary["best_epoch"] = epoch + 1

            wandb.save(best_model_path)

            print(f"Saved new best model: val_acc={val_acc:.4f}")

    wandb.finish()

if __name__ == "__main__":

    main()