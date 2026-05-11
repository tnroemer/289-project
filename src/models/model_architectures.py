import torch

from torch import nn
from torchvision import models

from models.lora import (
    LoRAConv2d,
    LoRALinear,
    count_trainable_parameters,
    freeze_all,
    inject_lora,
)


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

    raise ValueError(f"Unknown segmentation model: {model_name}")


# ============================================================================
# LeNet-5 (from-scratch small-capacity baseline).
#
# The original LeNet-5 expected 32x32 grayscale input; our pipeline serves
# 224x224 RGB. We keep LeNet-5's body identical and insert an
# AdaptiveAvgPool2d((5, 5)) before the FC stack so the linear dims stay sane
# regardless of input size. The conv channels (6 -> 16) and FC sizes
# (120 -> 84 -> num_classes) match the original paper.
# ============================================================================


class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.AdaptiveAvgPool2d((5, 5)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================================
# Pretrained torchvision backbones with two finetune variants:
#   *_head : freeze every original parameter, replace classifier head with a
#            fresh trainable nn.Linear, train ONLY the new head.
#   *_lora : freeze every original parameter, inject LoRA A/B pairs onto a
#            backbone-specific set of Conv2d/Linear layers, plus the fresh
#            head. The head + LoRA params are the only trainable tensors.
#
# All builders set `model.freeze_backbone = True` so the shared trainer keeps
# every BatchNorm in eval() mode for the whole run (matches the existing
# pretrained_resnet50 convention).
# ============================================================================


_LORA_DEFAULTS = {"rank": 8, "alpha": 16, "dropout": 0.0}


def _lora_kwargs(config):
    cfg = config or {}
    return {
        "rank": cfg.get("lora_rank", _LORA_DEFAULTS["rank"]),
        "alpha": cfg.get("lora_alpha", _LORA_DEFAULTS["alpha"]),
        "dropout": cfg.get("lora_dropout", _LORA_DEFAULTS["dropout"]),
    }


def _load_torchvision_model(loader_name, weights_enum_name, use_pretrained, **kwargs):
    loader = getattr(models, loader_name)
    try:
        if hasattr(models, weights_enum_name):
            weights = getattr(models, weights_enum_name).DEFAULT if use_pretrained else None
            return loader(weights=weights, **kwargs)
        return loader(pretrained=use_pretrained, **kwargs)
    except Exception as error:
        if use_pretrained:
            raise RuntimeError(
                f"Could not load pretrained weights for {loader_name}. "
                "Make sure the weights are cached on the compute node before training."
            ) from error
        try:
            return loader(weights=None, **kwargs)
        except TypeError:
            return loader(pretrained=False, **kwargs)


def _inject_lora_into_submodules(model, paths, config):
    kwargs = _lora_kwargs(config)
    total = 0
    for path in paths:
        submodule = model.get_submodule(path)
        total += inject_lora(submodule, rank=kwargs["rank"], alpha=kwargs["alpha"], dropout=kwargs["dropout"])
    return total


def _inject_lora_into_modules(parent, attrs, config):
    """LoRA-wrap specific named children of `parent` in place. Used for the
    alexnet/vgg16 case where we want to LoRA individual conv / fc layers
    rather than a whole submodule subtree."""
    kwargs = _lora_kwargs(config)
    total = 0
    for attr in attrs:
        child = parent[attr] if isinstance(attr, int) else getattr(parent, attr)
        if isinstance(child, nn.Conv2d):
            new = LoRAConv2d(child, **kwargs)
        elif isinstance(child, nn.Linear):
            new = LoRALinear(child, **kwargs)
        else:
            raise TypeError(f"Cannot LoRA-wrap {type(child).__name__} at {attr}")
        if isinstance(parent, nn.Sequential) or isinstance(attr, int):
            parent[attr] = new
        else:
            setattr(parent, attr, new)
        total += 1
    return total


def build_pretrained_resnet18(num_classes, mode, use_pretrained=True, config=None):
    model = _load_torchvision_model("resnet18", "ResNet18_Weights", use_pretrained)
    freeze_all(model)

    if mode == "lora":
        _inject_lora_into_submodules(model, ["layer4"], config)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    model.freeze_backbone = True
    return model


def build_pretrained_alexnet(num_classes, mode, use_pretrained=True, config=None):
    model = _load_torchvision_model("alexnet", "AlexNet_Weights", use_pretrained)
    freeze_all(model)

    if mode == "lora":
        # Last 2 conv layers in features + the 2 hidden FCs in classifier.
        # classifier[6] is the original 1000-class output; we replace it with
        # a fresh trainable head instead of LoRA-wrapping it.
        _inject_lora_into_modules(model.features, [8, 10], config)
        _inject_lora_into_modules(model.classifier, [1, 4], config)

    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)

    model.freeze_backbone = True
    return model


def build_pretrained_densenet121(num_classes, mode, use_pretrained=True, config=None):
    model = _load_torchvision_model("densenet121", "DenseNet121_Weights", use_pretrained)
    freeze_all(model)

    if mode == "lora":
        _inject_lora_into_submodules(model, ["features.denseblock4"], config)

    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    model.freeze_backbone = True
    return model


def build_pretrained_efficientnet_b0(num_classes, mode, use_pretrained=True, config=None):
    model = _load_torchvision_model("efficientnet_b0", "EfficientNet_B0_Weights", use_pretrained)
    freeze_all(model)

    if mode == "lora":
        # features[7] is the last MBConv stage, features[8] is the final 1x1.
        _inject_lora_into_submodules(model, ["features.7", "features.8"], config)

    # classifier is Sequential(Dropout, Linear).
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    model.freeze_backbone = True
    return model


def build_pretrained_convnext_tiny(num_classes, mode, use_pretrained=True, config=None):
    model = _load_torchvision_model("convnext_tiny", "ConvNeXt_Tiny_Weights", use_pretrained)
    freeze_all(model)

    if mode == "lora":
        # features[7] is the last stage (deepest CNBlocks).
        _inject_lora_into_submodules(model, ["features.7"], config)

    # classifier is Sequential(LayerNorm2d, Flatten, Linear).
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)

    model.freeze_backbone = True
    return model


def build_pretrained_vgg16(num_classes, mode, use_pretrained=True, config=None):
    model = _load_torchvision_model("vgg16", "VGG16_Weights", use_pretrained)
    freeze_all(model)

    if mode == "lora":
        # Last 2 conv layers (features[26], features[28]) + the 2 hidden FCs.
        _inject_lora_into_modules(model.features, [26, 28], config)
        _inject_lora_into_modules(model.classifier, [0, 3], config)

    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)

    model.freeze_backbone = True
    return model


def build_pretrained_googlenet(num_classes, mode, use_pretrained=True, config=None):
    # Load with aux_logits=True to match the pretrained weights, then disable
    # the aux outputs so forward() returns just the main logits (the trainer
    # expects a single logit tensor).
    model = _load_torchvision_model(
        "googlenet",
        "GoogLeNet_Weights",
        use_pretrained,
        aux_logits=True,
    )
    model.aux_logits = False
    freeze_all(model)

    if mode == "lora":
        _inject_lora_into_submodules(model, ["inception5a", "inception5b"], config)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    model.freeze_backbone = True
    return model


# Dispatch table: model_type -> (builder, mode) for the new pretrained backbones.
_PRETRAINED_BUILDERS = {
    "resnet18_head":          (build_pretrained_resnet18, "head"),
    "resnet18_lora":          (build_pretrained_resnet18, "lora"),
    "alexnet_head":           (build_pretrained_alexnet, "head"),
    "alexnet_lora":           (build_pretrained_alexnet, "lora"),
    "densenet121_head":       (build_pretrained_densenet121, "head"),
    "densenet121_lora":       (build_pretrained_densenet121, "lora"),
    "efficientnet_b0_head":   (build_pretrained_efficientnet_b0, "head"),
    "efficientnet_b0_lora":   (build_pretrained_efficientnet_b0, "lora"),
    "convnext_tiny_head":     (build_pretrained_convnext_tiny, "head"),
    "convnext_tiny_lora":     (build_pretrained_convnext_tiny, "lora"),
    "vgg16_head":             (build_pretrained_vgg16, "head"),
    "vgg16_lora":             (build_pretrained_vgg16, "lora"),
    "googlenet_head":         (build_pretrained_googlenet, "head"),
    "googlenet_lora":         (build_pretrained_googlenet, "lora"),
}


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

    if model_type == "lenet5":
        return LeNet5(num_classes=num_classes)

    if model_type in _PRETRAINED_BUILDERS:
        builder, mode = _PRETRAINED_BUILDERS[model_type]
        return builder(
            num_classes=num_classes,
            mode=mode,
            use_pretrained=config.get("use_pretrained_backbone", True),
            config=config,
        )

    raise ValueError(f"Unknown model type: {model_type}")
