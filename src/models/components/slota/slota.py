import torch
from torch import nn


class SlotAttention(nn.Module):
    """Slot Attention module.

    Args:
        num_slots: int - Number of slots in Slot Attention.
        iterations: int - Number of iterations in Slot Attention.
        num_attn_heads: int - Number of multi-head attention in Slot Attention,
    """

    def __init__(
        self,
        num_slots: int = 7,
        num_iterations: int = 3,
        num_attn_heads: int = 1,
        slot_dim: int = 64,
        hid_dim: int = 64,
        mlp_hid_dim: int = 128,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.num_attn_heads = num_attn_heads
        self.slot_dim = slot_dim
        self.hid_dim = hid_dim
        self.mlp_hid_dim = mlp_hid_dim
        self.eps = eps

        self.scale = (num_slots // num_attn_heads) ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, self.slot_dim))
        self.slots_sigma = nn.Parameter(torch.randn(1, 1, self.slot_dim))

        self.norm_input = nn.LayerNorm(self.hid_dim)
        self.norm_slot = nn.LayerNorm(self.slot_dim)
        self.norm_mlp = nn.LayerNorm(self.slot_dim)

        self.to_q = nn.Linear(self.slot_dim, self.slot_dim)
        self.to_k = nn.Linear(self.hid_dim, self.slot_dim)
        self.to_v = nn.Linear(self.hid_dim, self.slot_dim)

        self.gru = nn.GRUCell(self.slot_dim, self.slot_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.slot_dim, self.mlp_hid_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_hid_dim, self.slot_dim),
        )

    def forward(self, inputs, num_slots=None, train=False):
        outputs = dict()

        B, N_in, D_in = inputs.shape
        K = num_slots if num_slots is not None else self.num_slots
        D_slot = self.slot_dim
        N_heads = self.num_attn_heads

        mu = self.slots_mu.expand(B, K, -1)
        sigma = self.slots_sigma.expand(B, K, -1)
        slots = torch.normal(mu, torch.abs(sigma) + self.eps)

        inputs = self.norm_input(inputs)

        k = self.to_k(inputs).reshape(B, N_in, N_heads, -1).transpose(1, 2)
        v = self.to_v(inputs).reshape(B, N_in, N_heads, -1).transpose(1, 2)
        # k, v: (B, N_heads, N_in, D_slot // N_heads).

        if not train:
            attns = list()

        for iter_idx in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slot(slots)

            q = self.to_q(slots).reshape(B, K, N_heads, -1).transpose(1, 2)
            # q: (B, N_heads, K, slot_D // N_heads)

            attn_logits = torch.einsum("bhid, bhjd->bhij", k, q) * self.scale

            attn = attn_logits.softmax(dim=-1) + self.eps  # Normalization over slots
            # attn: (B, N_heads, N_in, K)

            if not train:
                attns.append(attn)

            attn = attn / torch.sum(attn, dim=-2, keepdim=True)  # Weighted mean
            # attn: (B, N_heads, N_in, K)

            updates = torch.einsum("bhij,bhid->bhjd", attn, v)
            # updates: (B, N_heads, K, slot_D // N_heads)
            updates = updates.transpose(1, 2).reshape(B, K, -1)
            # updates: (B, K, slot_D)

            slots = self.gru(updates.reshape(-1, D_slot), slots_prev.reshape(-1, D_slot))

            slots = slots.reshape(B, -1, D_slot)
            slots = slots + self.mlp(self.norm_mlp(slots))

        outputs["slots"] = slots
        outputs["attn"] = attn
        if not train:
            outputs["attns"] = torch.stack(attns, dim=1)
            # attns: (B, T, N_heads, N_in, K)

        return outputs


if __name__ == "__main__":
    _ = SlotAttention()
