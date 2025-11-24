import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from COCODataset import COCODataset
from model import CLIPModel  # <-- your model file
import torch.nn.functional as F

plot_file = 'loss.png'
save_file = 'weights.pth'

def clip_loss(image_embeds, text_embeds, temperature=0.07):
    # Normalize the embeddings
    image_embeds = F.normalize(image_embeds, dim=1)
    text_embeds  = F.normalize(text_embeds,  dim=1)

    # Cosine similarity matrix
    logits = (image_embeds @ text_embeds.T) / temperature

    batch_size = logits.size(0)
    labels = torch.arange(batch_size, device=logits.device)

    # Symmetric cross-entropy
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)

    return (loss_i + loss_t) / 2


def train_clip(
    coco_root="coco2014",
    cache_dir="cache",
    batch_size=64,
    lr=1e-4,
    epochs=5,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----------------------------------------------------
    # Load cached caption embeddings
    # ----------------------------------------------------
    cache_path = os.path.join(cache_dir, "train_caption_embeddings.pt")
    caption_cache = torch.load(cache_path)

    # ----------------------------------------------------
    # Image preprocessing
    # ----------------------------------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ])

    # ----------------------------------------------------
    # Dataset / Dataloader
    # ----------------------------------------------------
    train_images = f"{coco_root}/images/train2014"
    train_caps   = f"{coco_root}/annotations/captions_train2014.json"

    train_ds = COCODataset(
        img_dir=train_images,
        ann_file=train_caps,
        transform=transform,
        caption_cache=caption_cache,   # <--- enables embedding mode
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)

    # ----------------------------------------------------
    # Model & optimizer
    # ----------------------------------------------------
    model = CLIPModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # ----------------------------------------------------
    # Training loop
    # ----------------------------------------------------
    train_loss = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for images, text_embeds in train_loader:
            images = images.to(device)
            text_embeds = text_embeds.to(device)

            optimizer.zero_grad()

            # Get feature embeddings from your CLIP model
            _, image_features, text_features = model(images, text_embeds)

            # Compute InfoNCE loss
            loss = clip_loss(image_features, text_features)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        train_loss += [epoch_loss / len(train_loader)]
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f}")

    if save_file != None:
        torch.save(model.state_dict(), save_file)

    if plot_file != None:
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, label='avg Loss')
        #plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Progress')
        plt.savefig(plot_file)
        plt.close()

    print('Training complete! Model saved as', save_file)


if __name__ == "__main__":
    print('\t\tsave file = ', save_file)
    print('\t\tplot file = ', plot_file)
    train_clip()

'''import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from COCODataset import COCODataset

# --------------------------------------
# Load your custom CLIP model
# --------------------------------------
from clip_model import CLIPModel   # <-- Replace with your filename


# ================================================================
# 1. InfoNCE Loss (CLIP contrastive loss)
# ================================================================
class ClipInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_embeds, text_embeds):
        # Normalize embeddings
        image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)
        text_embeds  = text_embeds  / text_embeds.norm(dim=1, keepdim=True)

        # Cosine similarity matrix: (B × B)
        logits = (image_embeds @ text_embeds.T) / self.temperature

        batch_size = logits.size(0)
        labels = torch.arange(batch_size, device=logits.device)

        # Cross entropy on rows and columns
        loss_i = nn.functional.cross_entropy(logits, labels)
        loss_t = nn.functional.cross_entropy(logits.T, labels)

        return (loss_i + loss_t) / 2


# ================================================================
# 2. Dataset wrapper that loads cached caption embeddings
# ================================================================
class CLIPTrainDataset(torch.utils.data.Dataset):
    def __init__(self, coco_ds, caption_cache):
        self.coco_ds = coco_ds
        self.caption_cache = caption_cache

        # Map img_id → embedding
        self.caption_map = {c["image_id"]: c["embedding"] for c in caption_cache}

    def __len__(self):
        return len(self.coco_ds)

    def __getitem__(self, idx):
        image, caption, img_id = self.coco_ds[idx]
        text_emb = self.caption_map[img_id]  # torch.Tensor
        return image, text_emb


# ================================================================
# 3. Main training loop
# ================================================================
def train(
    coco_root="coco2014",
    cache_dir="cache",
    batch_size=64,
    num_epochs=5,
    lr=1e-4,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------
    # Data
    # --------------------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])

    train_images = f"{coco_root}/images/train2014"
    train_caps   = f"{coco_root}/annotations/captions_train2014.json"

    train_raw = COCODataset(train_images, train_caps, transform)

    cap_cache_path = f"{cache_dir}/train_caption_embeddings.pt"
    caption_cache = torch.load(cap_cache_path)

    train_ds = CLIPTrainDataset(train_raw, caption_cache)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)

    # --------------------------------------
    # Model, optimizer, loss
    # --------------------------------------
    model = CLIPModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = ClipInfoNCELoss(temperature=0.07)

    # --------------------------------------
    # Training Loop
    # --------------------------------------
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for images, text_embeds in train_loader:
            images = images.to(device)
            text_embeds = text_embeds.to(device)

            optimizer.zero_grad()

            # Forward through your CLIP model:
            # image_features: B × D
            # text_features:  B × D
            image_features, text_features = model(images, text_embeds)

            # InfoNCE contrastive loss
            loss = criterion(image_features, text_features)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "clip_model_trained.pt")
    print("Training complete. Saved clip_model_trained.pt")


# ================================================================
# Run training
# ================================================================
if __name__ == "__main__":
    train()
'''