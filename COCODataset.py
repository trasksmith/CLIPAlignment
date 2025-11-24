import os
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset

class COCODataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None, caption_cache=None):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.caption_cache = caption_cache

        # If a caption cache is provided, build a fast lookup dictionary
        if caption_cache is not None:
            self.caption_map = {
                item["image_id"]: item["embedding"]
                for item in caption_cache
            }

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

        # If we are training (cache exists), return cached text embedding
        if self.caption_cache is not None:
            text_emb = self.caption_map[img_id]
            return image, text_emb

        # Otherwise return caption string (for preprocessing)
        caption = self.coco.imgToAnns[img_id][0]["caption"]
        return image, caption, img_id
