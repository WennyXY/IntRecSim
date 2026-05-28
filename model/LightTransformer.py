import torch
import torch.nn as nn


class LightTransformerIntentModel(nn.Module):
    """Lightweight Transformer encoder for intent classification.

    Profile integration uses two complementary mechanisms, consistent with the
    GRU's h₀ philosophy of treating the profile as a *starting context*:

    1. **Profile bias on state tokens** — the profile is projected and added to
       every state token projection before the encoder.  Each state token is
       shifted into a profile-relative feature space so attention layers detect
       *deviations* rather than having to learn the profile through the CLS path.

    2. **Profile CLS token** — the profile is also prepended as position-0 token.
       After encoding, this token has attended to all state tokens and holds a
       compressed summary of how the recent sequence relates to the profile.
       Intent logits are read from this token's output.

    Architecture:
        state_proj         : state_dim   → hidden_dim        (per-step input projection)
        profile_shared     : profile_dim → hidden_dim        (shared backbone, tanh-gated)
        profile_cls_head   : hidden_dim  → hidden_dim        (CLS token from backbone)
        profile_bias_head  : hidden_dim  → hidden_dim        (additive state-token shift)
        pos_emb            : Embedding(max_seq_len+1, H)     (learned positional encoding)
        encoder            : num_layers × TransformerEncoderLayer  (pre-norm, batch_first)
        pre_head_norm      : LayerNorm(hidden_dim)
        intent_head        : hidden_dim → num_intents

    Pooling: logits are read from pre_head_norm(CLS_out + mean(state_out)).
    The additive combination lets the CLS path (profile-relative summary) and the
    mean-pool path (local temporal evidence) contribute independently.
    """

    def __init__(
        self,
        state_dim: int,
        profile_dim: int,
        hidden_dim: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        num_intents: int = 3,
        dropout: float = 0.0,
        max_seq_len: int = 16,
    ):
        super().__init__()
        if hidden_dim % nhead != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by nhead ({nhead})"
            )
        self.max_seq_len = max_seq_len

        # Shared profile backbone — projects once then branches into CLS token and
        # state-token bias.  Halves profile-related parameters vs. two independent
        # projections (saves profile_dim × hidden_dim, e.g. 384×256 = 98k for Beauty).
        self.profile_shared = nn.Linear(profile_dim, hidden_dim)
        self.profile_cls_head  = nn.Linear(hidden_dim, hidden_dim)      # CLS token
        # Bias head has no additive bias: the encoder's first pre-norm LayerNorm
        # centres the shifted tokens; a bias term here would be redundant.
        self.profile_bias_head = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.state_proj   = nn.Linear(state_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,    # pre-norm: more stable for small datasets
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_emb  = nn.Embedding(max_seq_len + 1, hidden_dim)  # +1 for CLS

        self.pre_head_norm = nn.LayerNorm(hidden_dim)
        self.head_drop     = nn.Dropout(dropout)
        self.intent_head   = nn.Linear(hidden_dim, num_intents)

    def forward(self, state_seq: torch.Tensor, static_profile: torch.Tensor) -> torch.Tensor:
        # state_seq:      (B, T, state_dim)
        # static_profile: (B, profile_dim)
        B, T, _ = state_seq.shape
        if T > self.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len {self.max_seq_len}")

        # Shared profile backbone — tanh keeps values bounded before branching,
        # consistent with the GRU/TCN profile-projection convention.
        profile_h = torch.tanh(self.profile_shared(static_profile))     # (B, H)

        # Shift every state token into a profile-relative feature space so the
        # encoder's attention layers focus on temporal deviations from the profile.
        profile_shift = self.profile_bias_head(profile_h).unsqueeze(1)  # (B, 1, H)
        x = self.state_proj(state_seq) + profile_shift                   # (B, T, H)

        # CLS token at position 0: after encoding it integrates all state tokens
        # and holds the intent-relevant summary.
        profile_token = self.profile_cls_head(profile_h).unsqueeze(1)   # (B, 1, H)
        tokens = torch.cat([profile_token, x], dim=1)                   # (B, T+1, H)

        positions = torch.arange(T + 1, device=state_seq.device)
        tokens = tokens + self.pos_emb(positions)

        out = self.encoder(tokens)                                       # (B, T+1, H)

        # Combine CLS token (global profile-relative summary) with mean-pooled
        # state tokens (local temporal evidence) — more robust than CLS alone on
        # short sequences and small datasets.
        cls_out   = out[:, 0]                                            # (B, H)
        mean_out  = out[:, 1:].mean(dim=1)                              # (B, H)
        pooled    = self.pre_head_norm(cls_out + mean_out)              # (B, H)
        return self.intent_head(self.head_drop(pooled))                  # (B, num_intents)
