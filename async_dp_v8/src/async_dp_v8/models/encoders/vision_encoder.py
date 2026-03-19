"""ResNet18-based vision encoder for wrist images."""
import torch
import torch.nn as nn
import torchvision.models as tvm


class VisionEncoder(nn.Module):
    def __init__(self, out_dim: int = 256, train_last_blocks: int = 2):
        super().__init__()
        backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        layers = list(backbone.children())[:-1]  # Remove FC
        self.backbone = nn.Sequential(*layers)
        self.proj = nn.Linear(512, out_dim)

        # Freeze all but last N blocks
        for p in self.backbone.parameters():
            p.requires_grad = False
        if train_last_blocks > 0:
            # Unfreeze last blocks (layer3, layer4 are indices 6, 7 in children)
            trainable_start = max(0, len(list(self.backbone.children())) - train_last_blocks - 1)
            for i, child in enumerate(self.backbone.children()):
                if i >= trainable_start:
                    for p in child.parameters():
                        p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B*T, 3, H, W] -> [B*T, out_dim]"""
        feat = self.backbone(x).flatten(1)
        return self.proj(feat)
