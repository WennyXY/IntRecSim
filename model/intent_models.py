import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from model.GRU import GRUIntentModel

from model.LightTransformer import LightTransformerIntentModel
from model.TCN import TCNIntentModel


MODEL_REGISTRY = {
    "gru": GRUIntentModel,
    "transformer": LightTransformerIntentModel,
    "tcn": TCNIntentModel,
}

_MODEL_KWARGS = {
    "gru": {"hidden_dim", "num_layers", "dropout"},
    "transformer": {"hidden_dim", "nhead", "num_layers", "dropout", "max_seq_len"},
    "tcn": {"hidden_dim", "num_layers", "dropout", "kernel_size"},
}


def compute_loss(intent_logits, intent_label, weight=None, label_smoothing=0.0):
    """Cross-entropy loss with optional class weighting and label smoothing.

    Args:
        intent_logits: (B, num_intents) raw logits from the model.
        intent_label:  (B,) integer class indices.
        weight:        (num_intents,) tensor of per-class weights.  Pass the
                       inverse-frequency weights computed from the training split
                       to counter class imbalance (stay_profile dominates).
        label_smoothing: Smoothing coefficient in [0, 1).  A value of 0.1 spreads
                       10 % of the probability mass uniformly across all classes,
                       reducing over-confidence on ambiguous intent boundaries.
    """
    return F.cross_entropy(
        intent_logits,
        intent_label,
        weight=weight,
        label_smoothing=label_smoothing,
    )

def build_intent_model(model_type, state_dim, profile_dim, num_intents, **kwargs):
    """Instantiate an intent model by name, passing only its accepted kwargs."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type {model_type!r}. Choose from {list(MODEL_REGISTRY)}"
        )
    accepted = _MODEL_KWARGS[model_type]
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return MODEL_REGISTRY[model_type](
        state_dim=state_dim,
        profile_dim=profile_dim,
        num_intents=num_intents,
        **filtered,
    )


class TransitionDataset(Dataset):
    """Dataset backed by state-transition .npz arrays with optional embedding lookup.

    When ``emb_dir`` is provided and the npz contains ``window_item_ids``,
    the dataset loads shared embedding tables (``items.npy``, ``profiles.npy``)
    into RAM and reconstructs per-sample embeddings via integer index lookups in
    ``__getitem__``.  This avoids storing large float arrays per sample on disk.

    Extra keys returned when embeddings are available:
        ``profile_emb``      : (D,)    — user's genre profile embedding
        ``recent_state_emb`` : (T, D)  — L2-normalised weighted mean of window
                                         item embeddings per timestep
        ``target_emb``       : (D,)    — target item embedding
    """

    def __init__(self, npz_data, emb_dir=None, decay_factor: float = 1.0):
        self.static_profile = torch.as_tensor(npz_data["static_profile"], dtype=torch.float32)
        self.state_sequence = torch.as_tensor(npz_data["state_sequence"], dtype=torch.float32)
        self.intent_label   = torch.as_tensor(npz_data["intent_label"],   dtype=torch.long)

        self.use_emb = False
        if emb_dir is not None and "window_item_ids" in npz_data:
            emb_dir = Path(emb_dir)
            # Embedding tables are small enough to keep fully in RAM (float32).
            self.item_embs    = torch.as_tensor(
                np.load(emb_dir / "items.npy").astype(np.float32)
            )  # (N_items, D)
            self.profile_embs = torch.as_tensor(
                np.load(emb_dir / "profiles.npy").astype(np.float32)
            )  # (N_profiles, D)
            self.window_item_ids = torch.as_tensor(
                npz_data["window_item_ids"], dtype=torch.long
            )  # (N, T, W)
            self.target_item_id  = torch.as_tensor(
                npz_data["target_item_id"],  dtype=torch.long
            )  # (N,)
            self.profile_emb_idx = torch.as_tensor(
                npz_data["profile_emb_idx"], dtype=torch.long
            )  # (N,)
            self._decay_w = self._make_decay_weights(
                decay_factor, self.window_item_ids.shape[-1]
            )
            self.use_emb = True

    @staticmethod
    def _make_decay_weights(decay_factor: float, window_size: int):
        """Pre-compute normalised decay weights over the window dimension."""
        if decay_factor == 1.0:
            return None
        w = torch.tensor(
            [decay_factor ** (window_size - 1 - i) for i in range(window_size)],
            dtype=torch.float32,
        )
        return w / w.sum()  # (W,)

    def __len__(self):
        return len(self.intent_label)

    def __getitem__(self, idx):
        item = {
            "state_seq": self.state_sequence[idx],
            "profile":   self.static_profile[idx],
            "intent":    self.intent_label[idx],
        }
        if self.use_emb:
            window_ids  = self.window_item_ids[idx]        # (T, W)
            window_embs = self.item_embs[window_ids]        # (T, W, D)
            if self._decay_w is None:
                recent_emb = window_embs.mean(dim=1)        # (T, D)
            else:
                recent_emb = (window_embs * self._decay_w[None, :, None]).sum(dim=1)
            item["profile_emb"]      = self.profile_embs[self.profile_emb_idx[idx]]   # (D,)
            item["recent_state_emb"] = F.normalize(recent_emb, dim=-1)                 # (T, D)
            item["target_emb"]       = self.item_embs[self.target_item_id[idx]]        # (D,)
        return item

