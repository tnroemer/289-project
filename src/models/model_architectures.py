import torch

from torch import nn
from torchvision import models


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class BasicCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 32, 0.05),
            ConvBlock(32, 64, 0.10),
            ConvBlock(64, 128, 0.15),
            ConvBlock(128, 256, 0.20),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.40),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


class BasicVIT(nn.Module):
    def __init__(
        self,
        num_classes,
        image_size=224,
        patch_size=16,
        embed_dim=192,
        depth=6,
        num_heads=6,
        mlp_dim=384,
        dropout=0.15,
    ):
        super().__init__()

        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_patches = (image_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.patch_norm = nn.LayerNorm(embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
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

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.patch_embed(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        x = self.patch_norm(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        x = self.transformer(x)
        x = x[:, 0]
        x = self.classifier(x)

        return x


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out


class SimpleResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(32, 32, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(32, 64, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(128, 256, num_blocks=2, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(256, num_classes),
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [ResNetBlock(in_channels, out_channels, stride=stride)]

        for _ in range(num_blocks - 1):
            layers.append(ResNetBlock(out_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


class UNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=32):
        super().__init__()

        self.down1 = UNetConvBlock(in_channels, base_channels)
        self.down2 = UNetConvBlock(base_channels, base_channels * 2)
        self.down3 = UNetConvBlock(base_channels * 2, base_channels * 4)
        self.bottleneck = UNetConvBlock(base_channels * 4, base_channels * 8)

        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.conv3 = UNetConvBlock(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.conv2 = UNetConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.conv1 = UNetConvBlock(base_channels * 2, base_channels)
        self.out = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(self.pool(down1))
        down3 = self.down3(self.pool(down2))
        bottleneck = self.bottleneck(self.pool(down3))

        x = self.up3(bottleneck)
        x = torch.cat((x, down3), dim=1)
        x = self.conv3(x)

        x = self.up2(x)
        x = torch.cat((x, down2), dim=1)
        x = self.conv2(x)

        x = self.up1(x)
        x = torch.cat((x, down1), dim=1)
        x = self.conv1(x)

        return self.out(x)


def build_pretrained_resnet50(num_classes, use_pretrained=True):
    try:
        if hasattr(models, "ResNet50_Weights"):
            weights = models.ResNet50_Weights.DEFAULT if use_pretrained else None
            model = models.resnet50(weights=weights)
        else:
            model = models.resnet50(pretrained=use_pretrained)
    except Exception as error:
        if use_pretrained:
            raise RuntimeError(
                "Could not load ImageNet ResNet50 weights. "
                "Make sure the weights are available on the compute server before training."
            ) from error

        try:
            model = models.resnet50(weights=None)
        except TypeError:
            model = models.resnet50(pretrained=False)

    for parameter in model.parameters():
        parameter.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.freeze_backbone = True

    return model


def build_deeplabv3_resnet50(out_channels=1, use_pretrained=True, aux_loss=True):
    try:
        if hasattr(models.segmentation, "DeepLabV3_ResNet50_Weights"):
            weights = models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT if use_pretrained else None
            model = models.segmentation.deeplabv3_resnet50(
                weights=weights,
                weights_backbone=None,
                aux_loss=aux_loss,
            )
        else:
            model = models.segmentation.deeplabv3_resnet50(pretrained=use_pretrained, aux_loss=aux_loss)
    except Exception as error:
        if use_pretrained:
            raise RuntimeError(
                "Could not load pretrained DeepLabV3-ResNet50 weights. "
                "Make sure the weights are available on the compute server before training."
            ) from error

        try:
            model = models.segmentation.deeplabv3_resnet50(
                weights=None,
                weights_backbone=None,
                aux_loss=aux_loss,
            )
        except TypeError:
            model = models.segmentation.deeplabv3_resnet50(pretrained=False, aux_loss=aux_loss)

    classifier_in_channels = model.classifier[-1].in_channels
    model.classifier[-1] = nn.Conv2d(classifier_in_channels, out_channels, kernel_size=1)

    if model.aux_classifier is not None:
        aux_in_channels = model.aux_classifier[-1].in_channels
        model.aux_classifier[-1] = nn.Conv2d(aux_in_channels, out_channels, kernel_size=1)

    return model


def build_segmentation_model(config=None):
    config = config or {}
    model_name = config.get("model", "DeepLabV3ResNet50")

    if model_name == "DeepLabV3ResNet50":
        return build_deeplabv3_resnet50(
            out_channels=config.get("out_channels", 1),
            use_pretrained=config.get("use_pretrained_backbone", False),
            aux_loss=config.get("aux_loss", True),
        )

    if model_name == "UNet":
        return UNet(
            in_channels=config.get("in_channels", 3),
            out_channels=config.get("out_channels", 1),
            base_channels=config.get("base_channels", 32),
        )

    raise ValueError(f"Unknown segmentation model: {model_name}")


def build_model(model_type, num_classes, config=None):
    config = config or {}

    if model_type == "cnn":
        return BasicCNN(num_classes=num_classes)

    if model_type == "vit":
        return BasicVIT(
            num_classes=num_classes,
            image_size=config.get("image_size", 224),
            patch_size=config.get("patch_size", 16),
            embed_dim=config.get("embed_dim", 192),
            depth=config.get("depth", 6),
            num_heads=config.get("num_heads", 6),
            mlp_dim=config.get("mlp_dim", 384),
            dropout=config.get("dropout", 0.15),
        )

    if model_type == "resnet":
        return SimpleResNet(num_classes=num_classes)

    if model_type == "pretrained_resnet50":
        return build_pretrained_resnet50(
            num_classes=num_classes,
            use_pretrained=config.get("use_pretrained_backbone", False),
        )

    raise ValueError(f"Unknown model type: {model_type}")
