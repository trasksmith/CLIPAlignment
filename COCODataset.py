import os
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset

class COCODataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        self.img_dir = img_dir
        print("Loading COCO annotations...")
        self.coco = COCO(ann_file)
        print("COCO loaded. Number of images:", len(self.coco.imgs))
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform

    '''def __init__(self, img_dir, ann_file, transform=None, caption_cache_path=None):
        self.img_dir = img_dir
        print("Loading COCO annotations...")
        self.coco = COCO(ann_file)
        print("COCO loaded. Number of images:", len(self.coco.imgs))
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        print("Loading caption cache...")
        self.caption_cache = torch.load(caption_cache_path, map_location='cpu')
        print("Caption cache loaded.")

        # Only load caption cache if path is provided, but keep on CPU
        if caption_cache_path is not None:
            self.caption_cache = torch.load(caption_cache_path, map_location='cpu')
            # Build fast lookup dict if needed
            self.caption_map = {item['image_id']: item['embedding'] for item in self.caption_cache}
        else:
            self.caption_cache = None
            self.caption_map = None'''

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        # Load image
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Return caption string for on-the-fly embedding
        caption = self.coco.imgToAnns[img_id][0]["caption"]
        return image, caption
'''
    def __getitem__(self, idx):
        img_id = self.ids[idx]

        # Load image
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.caption_cache is not None:
            text_emb = self.caption_map[img_id]
            # Move embedding to GPU **in the training loop**, not here
            text_emb = torch.tensor(text_emb)  # ensure tensor type
            return image, text_emb

        # Otherwise return caption string (for preprocessing)
        caption = self.coco.imgToAnns[img_id][0]["caption"]
        return image, caption, img_id
'''

