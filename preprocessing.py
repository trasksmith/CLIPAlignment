import os
import torch
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel
from COCODataset import COCODataset

COCO_ROOT = "coco2014"

TRAIN_IMAGES = f"{COCO_ROOT}/images/train2014"
VAL_IMAGES   = f"{COCO_ROOT}/images/val2014"

TRAIN_CAPTIONS = f"{COCO_ROOT}/annotations/captions_train2014.json"
VAL_CAPTIONS   = f"{COCO_ROOT}/annotations/captions_val2014.json"

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                         (0.26862954, 0.26130258, 0.27577711)),
])

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
text_encoder.eval()

def encode_caption(caption):
    tokens = tokenizer(
        caption,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
    ).to(device)

    with torch.no_grad():
        out = text_encoder(**tokens)
        emb = out.last_hidden_state[:, 0, :]  # CLS token

    return emb.squeeze(0).cpu()

def build_cache(dataset, split_name):
    print(f"\nEncoding {split_name} captions...")
    cache = []

    for _, caption, img_id in dataset:
        emb = encode_caption(caption)
        cache.append({
            "image_id": img_id,
            "caption": caption,
            "embedding": emb,
        })

    out_path = f"{CACHE_DIR}/{split_name}_caption_embeddings.pt"
    torch.save(cache, out_path)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    print("Loading COCO datasets with your folder structure...")

    train_ds = COCODataset(TRAIN_IMAGES, TRAIN_CAPTIONS, transform)
    val_ds   = COCODataset(VAL_IMAGES,   VAL_CAPTIONS,   transform)

    print(f"Train split size: {len(train_ds)} images")
    print(f"Val split size:   {len(val_ds)} images")

    # Build caches
    build_cache(train_ds, "train")
    build_cache(val_ds, "val")

    print("\nPreprocessing complete.")