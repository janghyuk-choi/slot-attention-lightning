from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn

from src.models.components.slota import SlotAttention
from src.models.components.slota_utils import Decoder, Encoder


class SlotAttentionAutoEncoder(nn.Module):
    """Builds Slot Attention-based auto-encoder for object discovery.

    Args:
        num_slots (int): Number of slots in Slot Attention.
    """

    def __init__(
        self,
        img_size: int = 128,
        num_slots: int = 7,
        num_iterations: int = 3,
        num_attn_heads: int = 1,
        hid_dim: int = 64,
        slot_dim: int = 64,
        mlp_hid_dim: int = 128,
        eps: float = 1e-8,
        enc_depth: int = 4,
        dec_hid_dim: int = 64,
        dec_init_size: int = 8,
        dec_depth: int = 6,
    ):
        super().__init__()
        self.num_slots = num_slots

        self.encoder_cnn = Encoder(
            img_size=img_size,
            hid_dim=hid_dim,
            enc_depth=enc_depth,
        )
        self.decoder_cnn = Decoder(
            img_size=img_size,
            slot_dim=slot_dim,
            dec_hid_dim=dec_hid_dim,
            dec_init_size=dec_init_size,
            dec_depth=dec_depth,
        )

        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            num_iterations=num_iterations,
            num_attn_heads=num_attn_heads,
            slot_dim=slot_dim,
            hid_dim=hid_dim,
            mlp_hid_dim=mlp_hid_dim,
            eps=eps,
        )

    def forward(self, image, train=True):
        # `image`: (batch_size, num_channels, height, width)
        B, C, H, W = image.shape

        # Convolutional encoder with position embedding
        x = self.encoder_cnn(image)  # CNN Backbone
        # `x`: (B, height * width, hid_dim)

        # Slot Attention module.
        slota_outputs = self.slot_attention(x, train=train)
        slots = slota_outputs["slots"]
        # `slots`: (N, K, slot_dim)

        x = self.decoder_cnn(slots)
        # `x`: (B*K, height, width, num_channels+1)

        # Undo combination of slot and batch dimension; split alpha masks
        recons, masks = x.reshape(B, self.num_slots, H, W, C + 1).split([3, 1], dim=-1)
        # `recons`: (B, K, height, width, num_channels)
        # `masks`: (B, K, height, width, 1)

        # Normalize alpha masks over slots.
        masks = nn.Softmax(dim=1)(masks)

        recon_combined = torch.sum(recons * masks, dim=1)  # Recombine image
        recon_combined = recon_combined.permute(0, 3, 1, 2)
        # `recon_combined`: (batch_size, num_channels, height, width)

        outputs = dict()
        outputs["recon_combined"] = recon_combined
        outputs["recons"] = recons
        outputs["masks"] = masks
        outputs["slots"] = slots
        outputs["attn"] = slota_outputs["attn"]
        if not train:
            outputs["attns"] = slota_outputs["attns"]
            # `attns`: (B, T, N_heads, N_in, K)

        return outputs


if __name__ == "__main__":
    _ = SlotAttentionAutoEncoder()
