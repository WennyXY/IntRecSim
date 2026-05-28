import torch
import torch.nn as nn


class GRUIntentModel(nn.Module):
    """GRU classifier for preference-transition intent.

    The static profile initialises the GRU hidden state (one vector per layer)
    rather than being added as a constant bias to every input token.  This lets
    the model treat the profile as the user's *starting point* and learn how
    much the recent sequence deviates from it — a structurally better fit for
    modelling intent drift.

    Architecture:
        state_proj   : state_dim  → hidden_dim               (per-step input projection)
        profile_proj : profile_dim → hidden_dim * num_layers  (h_0 initialisation)
        gru          : hidden_dim → hidden_dim
        pre_head_norm: LayerNorm(hidden_dim)
        head_drop    : Dropout
        intent_head  : hidden_dim → num_intents
    """

    def __init__(
        self,
        state_dim,
        profile_dim,
        hidden_dim=128,
        num_layers=1,
        num_intents=3,
        dropout=0.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        gru_dropout = dropout if num_layers > 1 else 0.0
        self.state_proj   = nn.Linear(state_dim, hidden_dim)
        # Projects profile to all GRU layers' initial hidden states in one pass.
        # For num_layers=1 the output shape (hidden_dim) matches the old profile_proj,
        # but the semantics change: it becomes h_0 rather than an additive bias.
        self.profile_proj = nn.Linear(profile_dim, hidden_dim * num_layers)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.pre_head_norm = nn.LayerNorm(hidden_dim)
        self.head_drop     = nn.Dropout(dropout)
        self.intent_head   = nn.Linear(hidden_dim, num_intents)

    def forward(self, state_seq, static_profile):
        # state_seq: (B, T, state_dim)   static_profile: (B, profile_dim)
        B = state_seq.size(0)
        x = self.state_proj(state_seq)                                    # (B, T, H)

        # Initialise each GRU layer's hidden state from the static profile.
        # tanh keeps h_0 in the same range as a saturated GRU hidden state.
        h0 = torch.tanh(self.profile_proj(static_profile))               # (B, H * num_layers)
        h0 = h0.view(B, self.num_layers, self.hidden_dim)                # (B, num_layers, H)
        h0 = h0.permute(1, 0, 2).contiguous()                            # (num_layers, B, H)

        _, h = self.gru(x, h0)
        return self.intent_head(self.head_drop(self.pre_head_norm(h[-1])))  # (B, num_intents)
