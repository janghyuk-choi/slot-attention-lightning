import json
import os
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import transforms


class CLEVR6(Dataset):
    def __init__(
        self,
        data_dir: str = "data/clevr_with_masks/CLEVR6",
        img_size: int = 128,
        transform: transforms.Compose = None,
        train: bool = True,
    ):
        super().__init__()

        self.img_size = img_size
        self.train = train
        self.stage = "train" if train else "val"

        self.max_num_objs = 6
        self.max_num_masks = self.max_num_objs + 1

        self.image_dir = os.path.join(data_dir, "images", self.stage)
        self.mask_dir = os.path.join(data_dir, "masks", self.stage)
        self.scene_dir = os.path.join(data_dir, "scenes")
        self.metadata = json.load(
            open(os.path.join(self.scene_dir, f"CLEVR_{self.stage}_scenes.json"))
        )

        self.files = sorted(os.listdir(self.image_dir))
        self.num_files = len(self.files)

        self.transform = transform

        if not train:
            self.masks = defaultdict(list)
            masks = sorted(os.listdir(self.mask_dir))
            for mask in masks:
                split = mask.split("_")
                filename = "_".join(split[:3]) + ".png"
                self.masks[filename].append(mask)
            del masks

    def __getitem__(self, index):
        filename = self.metadata["scenes"][index]["image_filename"]

        img = (
            read_image(os.path.join(self.image_dir, filename), ImageReadMode.RGB)
            .float()
            .div(255.0)
        )
        img = self.transform(img)
        sample = {"image": img}

        if not self.train:
            masks = list()
            for mask_filename in self.masks[filename]:
                mask = (
                    read_image(os.path.join(self.mask_dir, mask_filename), ImageReadMode.GRAY)
                    .div(255)
                    .long()
                )
                mask = self.transform(mask)
                masks.append(mask)
            masks = torch.cat(masks, dim=0).unsqueeze(-1)
            # `masks`: (num_objects + 1, H, W, 1)

            num_masks = masks.shape[0]
            if num_masks < self.max_num_masks:
                masks = torch.cat(
                    (
                        masks,
                        torch.zeros(
                            (self.max_num_masks - num_masks, self.img_size, self.img_size, 1)
                        ),
                    ),
                    dim=0,
                )
            # `masks``: (max_num_masks, H, W, 1)

            sample["masks"] = masks.float()
            sample["num_objects"] = num_masks - 1

        return sample

    def __len__(self):
        return self.num_files
