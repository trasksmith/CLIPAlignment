import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()

        # 1. Pretrained ResNet50 backbone
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # remove the final classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # 2. Projection head: 2048 → 1024 → GELU → 512
        self.projection = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Linear(1024, embed_dim)
        )

    def forward(self, x):
        # ResNet output: (B, 2048, 1, 1)
        x = self.backbone(x)
        x = x.view(x.size(0), -1)          # (B, 2048)

        x = self.projection(x)             # (B, 512)
        x = x / x.norm(dim=-1, keepdim=True)  # normalize
        return x


class CLIPModel(nn.Module):
    """
    This version of CLIP takes:
        - images
        - precomputed *cached caption embeddings*

    The text encoder is NOT used (frozen and removed from forward).
    """
    def __init__(self, embed_dim=512):
        super().__init__()

        self.image_encoder = ImageEncoder(embed_dim)

        # logit scaling factor used by CLIP
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, images, caption_embeds):
        """
        images: (B, 3, 224, 224)
        caption_embeds: (B, 512) from your cache
        """

        # Encode images
        img_embeds = self.image_encoder(images)

        # Captions are already CLIP embeddings, but normalize again for safety
        caption_embeds = caption_embeds / caption_embeds.norm(dim=-1, keepdim=True)

        # Compute scaled cosine similarity
        scale = self.logit_scale.exp()
        logits = scale * img_embeds @ caption_embeds.t()  # (B, B)

        return logits, img_embeds, caption_embeds
