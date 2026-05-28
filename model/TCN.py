import torch
import torch.nn as nn


class _CausalResBlock(nn.Module):
    """Dilated causal convolution block with residual connection and LayerNorm.

    Dilation doubles each layer (2**i), giving the stack an exponentially
    growing receptive field without increasing parameter count.
    Residual path lets gradients bypass saturated activations in deeper nets.
    """

    def __init__(self, hidden_dim: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation          # left-only: causal
        self.pad  = nn.ConstantPad1d((padding, 0), 0)
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, dilation=dilation)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act  = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, L)
        out = self.conv(self.pad(x))           # (B, H, L)
        out = self.norm(out.transpose(1, 2))   # (B, L, H) — LayerNorm over H
        out = self.drop(self.act(out))
        return out.transpose(1, 2) + x         # (B, H, L) residual


class TCNIntentModel(nn.Module):
    """Dilated causal TCN with residual blocks for intent prediction.

    Profile integration follows the same *starting-context* philosophy as the GRU:
    the static profile is projected and added as a bias to the input sequence
    projection so every convolutional layer operates in a profile-relative feature
    space.  The temporal stack then learns how the recent window *deviates* from
    that context rather than having to re-discover the profile from a late-fusion
    concatenation.

    Architecture:
        state_proj   : state_dim  → hidden_dim          (per-step input projection)
        profile_proj : profile_dim → hidden_dim          (early profile bias)
        blocks       : num_layers × _CausalResBlock      (dilation = 1, 2, 4, …)
        norm         : LayerNorm on last-step output
        intent_head  : hidden_dim → num_intents
    """

    def __init__(
        self,
        state_dim: int,
        profile_dim: int,
        hidden_dim: int = 128,
        num_intents: int = 3,
        kernel_size: int = 3,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.state_proj   = nn.Linear(state_dim,   hidden_dim)
        # Projects profile to an additive input bias; tanh matches the GRU h₀
        # convention and keeps values in a bounded range before the first conv.
        self.profile_proj = nn.Linear(profile_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            _CausalResBlock(hidden_dim, kernel_size, dilation=2 ** i, dropout=dropout)
            for i in range(num_layers)
        ])

        self.norm        = nn.LayerNorm(hidden_dim)
        self.head_drop   = nn.Dropout(dropout)
        self.intent_head = nn.Linear(hidden_dim, num_intents)

    def forward(self, state_seq: torch.Tensor, static_profile: torch.Tensor) -> torch.Tensor:
        # state_seq:      (B, L, state_dim)
        # static_profile: (B, profile_dim)

        # Shift input sequence into a profile-relative feature space.
        profile_bias = torch.tanh(self.profile_proj(static_profile)).unsqueeze(1)  # (B, 1, H)
        x = self.state_proj(state_seq) + profile_bias                              # (B, L, H)

        x = x.transpose(1, 2)          # (B, H, L) — Conv1d is channel-first
        for block in self.blocks:
            x = block(x)

        h = self.norm(x[:, :, -1])     # (B, H) — most recent (causal) timestep
        return self.intent_head(self.head_drop(h))   # (B, num_intents)
