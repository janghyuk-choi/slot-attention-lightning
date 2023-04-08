import json
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import transforms


class CLEVRTEX(Dataset):
    def __init__(
        self,
        data_dir: str = "data/clevrtex",
        img_size: int = 128,
        transform: transforms.Compose = None,
        train: bool = True,
    ):
        super().__init__()

        self.img_size = img_size
        self.train = train
        self.stage = "train" if train else "val"

        self.max_num_objs = 10
        self.max_num_masks = self.max_num_objs + 1

        self.image_dir = os.path.join(data_dir, "images", self.stage)
        self.mask_dir = os.path.join(data_dir, "masks", self.stage)
        self.scene_dir = os.path.join(data_dir, "scenes")

        self.files = sorted(os.listdir(self.image_dir))
        self.num_files = len(self.files)

        self.transform = transform

    def __getitem__(self, index):
        image_filename = self.files[index]
        img = (
            read_image(os.path.join(self.image_dir, image_filename), ImageReadMode.RGB)
            .float()
            .div(255.0)
        )
        img = self.transform(img)
        sample = {"image": img}

        scene_name = image_filename[:-3] + "json"
        metadata = json.load(open(os.path.join(self.scene_dir, self.stage, scene_name)))

        if not self.train:
            mask_filename = image_filename[:-4] + "_flat.png"
            masks = read_image(os.path.join(self.mask_dir, mask_filename)).long().squeeze(0)
            masks = F.one_hot(masks, self.max_num_masks).permute(2, 0, 1)
            masks = self.transform(masks).unsqueeze(-1)
            # masks: (max_num_masks, H, W, 1)

            sample["masks"] = masks.float()
            sample["num_objects"] = len(metadata["objects"])

        return sample

    def __len__(self):
        return self.num_files
