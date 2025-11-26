import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from COCODataset import COCODataset
from model import CLIPModel  # <-- your model file
from transformers import CLIPTokenizer, CLIPTextModel
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

def encode_caption_batch(captions):
    """
    captions: list of strings
    returns: normalized tensor of shape (B, embed_dim)
    """
    tokens = tokenizer(
        captions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length
    ).to(device)

    with torch.no_grad():
        out = text_encoder(**tokens)
        embeddings = out.last_hidden_state[:, 0, :]  # CLS token

    embeddings = F.normalize(embeddings, dim=-1)
    return embeddings

def train_clip(
    coco_root="coco2014",
    cache_dir="cache",
    batch_size=32,
    lr=1e-4,
    epochs=10,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----------------------------------------------------
    # Load cached caption embeddings
    # ----------------------------------------------------
    #train_cache_path = os.path.join(cache_dir, "train_caption_embeddings.pt")
    #val_cache_path = os.path.join(cache_dir, "val_caption_embeddings.pt")
    #train_caption_cache = torch.load(train_cache_path)
    #val_caption_cache = torch.load(val_cache_path)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    text_encoder.eval()  # freeze weights, we only train image encoder

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

    val_images = f"{coco_root}/images/val2014"
    val_caps = f"{coco_root}/annotations/captions_val2014.json"

    train_dataset = COCODataset(
        img_dir=train_images,
        ann_file=train_caps,
        transform=transform,
        #caption_cache_path=train_cache_path   # <--- enables embedding mode
    )

    val_dataset = COCODataset(
        img_dir=val_images,
        ann_file=val_caps,
        transform=transform,
        #caption_cache_path=val_cache_path
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ----------------------------------------------------
    # Model & optimizer
    # ----------------------------------------------------
    model = CLIPModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ----------------------------------------------------
    # Training loop
    # ----------------------------------------------------
    train_loss, val_loss = [], []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0

        for images, captions in train_loader:
            images = images.to(device)

            '''if not isinstance(text_embeds, torch.Tensor):
                text_embeds = torch.tensor(text_embeds, dtype=torch.float32)
            text_embeds = text_embeds.to(device)'''

            # Compute text embeddings dynamically
            text_embeds = encode_caption_batch(captions)

            # Compute image embeddings
            img_embeds, text_embeds = model(images, text_embeds)

            optimizer.zero_grad()

            # Get feature embeddings from your CLIP model
            #image_features, text_features = model(images, text_embeds)

            # Compute InfoNCE loss
            loss = clip_loss(img_embeds, text_embeds)

            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        train_loss += [epoch_train_loss / len(train_loader)]


        model.eval()
        epoch_val_loss = 0.0

        with torch.no_grad():
            for images, captions in val_loader:
                images = images.to(device)

                text_embeds = encode_caption_batch(captions)
                img_embeds, text_embeds = model(images, text_embeds)

                #image_features, text_features = model(images, text_embeds)
                loss = clip_loss(img_embeds, text_embeds)
                epoch_val_loss += loss.item()

        val_loss += [epoch_val_loss / len(val_loader)]

        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss[-1]:.4f} | Val Loss: {val_loss[-1]:.4f}")

    if save_file != None:
        torch.save(model.state_dict(), save_file)

    if plot_file != None:
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, label='avg Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Progress')
        plt.savefig(plot_file)
        plt.close()

    print('Training complete! Model saved as', save_file)
    print('Plotting complete! Plot saved as', plot_file)


if __name__ == "__main__":
    print('\t\tsave file = ', save_file)
    print('\t\tplot file = ', plot_file)
    train_clip()