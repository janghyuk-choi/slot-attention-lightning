from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn

class SoftPositionEmbed(nn.Module):
    """Builds the soft position embedding layer with learnable projection.

    Args:
        hid_dim (int): Size of input feature dimension.
        resolution (tuple): Tuple of integers specifying width and height of grid.
    """

    def __init__(
        self,
        hid_dim: int = 64,
        resolution: Tuple[int, int] = (128, 128),
    ):
        super().__init__()
        self.embedding = nn.Linear(4, hid_dim, bias=True)
        self.grid = self.build_grid(resolution)

    def forward(self, inputs):
        self.grid = self.grid.to(inputs.device)
        grid = self.embedding(self.grid).to(inputs.device)
        return inputs + grid

    def build_grid(self, resolution):
        ranges = [np.linspace(0.0, 1.0, num=res) for res in resolution]
        grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
        grid = np.stack(grid, axis=-1)
        grid = np.reshape(grid, [resolution[0], resolution[1], -1])
        grid = np.expand_dims(grid, axis=0)
        grid = grid.astype(np.float32)
        return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1))


class Encoder(nn.Module):
    def __init__(
        self,
        img_size: int = 128,
        hid_dim: int = 64,
        enc_depth: int = 4,
    ):
        super().__init__()
        assert enc_depth > 2, "Depth must be larger than 2."

        convs = nn.ModuleList([nn.Conv2d(3, hid_dim, 5, padding="same"), nn.ReLU()])
        for _ in range(enc_depth - 2):
            convs.extend([nn.Conv2d(hid_dim, hid_dim, 5, padding="same"), nn.ReLU()])
        convs.append(nn.Conv2d(hid_dim, hid_dim, 5, padding="same"))
        self.convs = nn.Sequential(*convs)

        self.encoder_pos = SoftPositionEmbed(hid_dim, (img_size, img_size))
        self.layer_norm = nn.LayerNorm([img_size * img_size, hid_dim])
        self.mlp = nn.Sequential(
            nn.Linear(hid_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, hid_dim)
        )

    def forward(self, x):
        x = self.convs(x)  # [B, D, H, W]
        x = x.permute(0, 2, 3, 1)  # [B, H, W ,D]
        x = self.encoder_pos(x)
        x = torch.flatten(x, 1, 2)
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        img_size: int = 128,
        slot_dim: int = 64,
        dec_hid_dim: int = 64,
        dec_init_size: int = 8,
        dec_depth: int = 6,
    ):
        super().__init__()

        self.img_size = img_size
        self.dec_init_size = dec_init_size
        self.decoder_pos = SoftPositionEmbed(slot_dim, (dec_init_size, dec_init_size))

        D_slot = slot_dim
        D_hid = dec_hid_dim
        upsample_step = int(np.log2(img_size // dec_init_size))

        deconvs = nn.ModuleList()
        count_layer = 0
        for _ in range(upsample_step):
            deconvs.extend(
                [
                    nn.ConvTranspose2d(
                        D_hid if count_layer > 0 else D_slot,
                        D_hid,
                        5,
                        stride=(2, 2),
                        padding=2,
                        output_padding=1,
                    ),
                    nn.ReLU(),
                ]
            )
            count_layer += 1

        for _ in range(dec_depth - upsample_step - 1):
            deconvs.extend(
                [
                    nn.ConvTranspose2d(
                        D_hid if count_layer > 0 else D_slot, D_hid, 5, stride=(1, 1), padding=2
                    ),
                    nn.ReLU(),
                ]
            )
            count_layer += 1

        deconvs.append(nn.ConvTranspose2d(D_hid, 4, 3, stride=(1, 1), padding=1))
        self.deconvs = nn.Sequential(*deconvs)

    def forward(self, x):
        """Broadcast slot features to a 2D grid and collapse slot dimension."""
        x = x.reshape(-1, x.shape[-1]).unsqueeze(1).unsqueeze(2)
        x = x.repeat((1, self.dec_init_size, self.dec_init_size, 1))
        x = self.decoder_pos(x)
        x = x.permute(0, 3, 1, 2)
        x = self.deconvs(x)
        x = x[:, :, : self.img_size, : self.img_size]
        x = x.permute(0, 2, 3, 1)
        return x