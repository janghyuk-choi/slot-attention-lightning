import json
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from pycocotools import mask as pycocotools_mask
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import transforms


class PTR(Dataset):
    def __init__(
        self,
        data_dir: str = "data/PTR",
        img_size: int = 128,
        transform: transforms.Compose = None,
        train: bool = True,
    ):
        super().__init__()

        self.max_num_objs = 6
        self.max_num_masks = self.max_num_objs + 1

        self.img_size = img_size
        self.train = train
        self.stage = "train" if train else "val"

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
            masks = list()
            for obj in metadata["objects"]:
                masks.append(obj["obj_mask"])
            masks = torch.tensor(pycocotools_mask.decode(masks), dtype=torch.long)
            masks = torch.einsum("hwn -> nhw", masks)
            masks = self.transform(masks.unsqueeze(1)).squeeze(1)
            masks = torch.cat(
                [(torch.sum(masks, dim=0, keepdim=True) == 0).long(), masks], dim=0
            ).unsqueeze(-1)
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
