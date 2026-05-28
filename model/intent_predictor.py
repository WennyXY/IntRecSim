import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model.intent_models import build_intent_model


INTENT_ID2NAME = {
    0: "stay_profile",
    1: "follow_recent",
    2: "explore",
}

INTENT_BIAS = {
    "stay_profile":  {"profile": 0.70, "recent": 0.20, "novelty": 0.10},
    "follow_recent": {"profile": 0.20, "recent": 0.70, "novelty": 0.10},
    "explore":       {"profile": 0.15, "recent": 0.15, "novelty": 0.70},
}

_CKPT_NAMES = ["best_intent_model.pt", "best_gru_intent.pt"]


class IntentPredictor:
    """Model-agnostic intent predictor loaded from a training checkpoint directory.

    Supports any model registered in ``model.intent_models.MODEL_REGISTRY``.
    The model type is read from ``train_config.json``; defaults to ``"gru"``
    for checkpoints produced before ``model_type`` was saved.

    Usage::
        predictor = IntentPredictor("output/state_transition/ml_1m/gru/ckpt_20260505_1234")
        intent_name, bias = predictor.predict(state_sequence, static_profile)
    """

    def __init__(self, ckpt_dir):
        ckpt_dir = Path(ckpt_dir)

        with (ckpt_dir / "train_config.json").open() as f:
            cfg = json.load(f)

        self.model = build_intent_model(
            model_type=cfg.get("model_type", "gru"),
            state_dim=cfg["state_dim"],
            profile_dim=cfg["profile_dim"],
            num_intents=cfg["num_intents"],
            hidden_dim=cfg.get("hidden_dim", 128),
            num_layers=cfg.get("num_layers", 1),
            dropout=0.0,
            nhead=cfg.get("nhead", 4),
            max_seq_len=cfg.get("max_seq_len", 16),
            kernel_size=cfg.get("kernel_size", 3),
        )

        ckpt_path = self._find_checkpoint(ckpt_dir)
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location="cpu")

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self._device = torch.device("cpu")

        self.val_acc      = ckpt.get("val_acc")
        self.val_macro_f1 = ckpt.get("val_macro_f1")
        self.epoch        = ckpt.get("epoch")
        self.model_type   = cfg.get("model_type", "gru")
        self.profile_dim  = cfg["profile_dim"]
        self.state_dim    = cfg["state_dim"]
        self.input_repr   = cfg.get("input_repr", "auto")
        self.decay_factor = float(cfg.get("decay_factor", 1.0))
        self.emb_dir      = cfg.get("emb_dir")  # str or None

    @staticmethod
    def _find_checkpoint(ckpt_dir):
        for name in _CKPT_NAMES:
            path = ckpt_dir / name
            if path.exists():
                return path
        raise FileNotFoundError(
            f"No checkpoint found in {ckpt_dir}. Tried: {_CKPT_NAMES}"
        )

    def to(self, device):
        self._device = torch.device(device)
        self.model.to(self._device)
        return self

    @torch.no_grad()
    def _forward(self, state_sequence, static_profile):
        """Single forward pass. Returns (probs, single) where probs is (B, num_intents)."""
        state_seq = torch.as_tensor(np.asarray(state_sequence), dtype=torch.float32)
        profile   = torch.as_tensor(np.asarray(static_profile),  dtype=torch.float32)
        single = state_seq.dim() == 2
        if single:
            state_seq = state_seq.unsqueeze(0)
            profile   = profile.unsqueeze(0)
        logits = self.model(state_seq.to(self._device), profile.to(self._device))
        probs  = F.softmax(logits, dim=-1).cpu().numpy()
        return probs, single

    def predict(self, state_sequence, static_profile):
        """Predict intent for one sample or a batch.

        Args:
            state_sequence: (seq_len, state_dim) or (B, seq_len, state_dim).
            static_profile: (profile_dim,) or (B, profile_dim).

        Returns:
            Single input: (intent_name: str, bias_vector: dict).
            Batch input: (intent_names: list[str], bias_vectors: list[dict]).
        """
        probs, single = self._forward(state_sequence, static_profile)
        pred_ids = probs.argmax(axis=-1).tolist()
        if single:
            name = INTENT_ID2NAME[pred_ids[0]]
            return name, INTENT_BIAS[name].copy()
        names = [INTENT_ID2NAME[i] for i in pred_ids]
        return names, [INTENT_BIAS[n].copy() for n in names]

    def predict_proba(self, state_sequence, static_profile):
        """Return softmax probabilities over intents.

        Returns:
            numpy array of shape (num_intents,) for single input,
            or (B, num_intents) for batch input.
        """
        probs, single = self._forward(state_sequence, static_profile)
        return probs[0] if single else probs

    def predict_with_confidence(self, state_sequence, static_profile):
        """Predict intent and return confidence in one forward pass.

        Confidence is the softmax probability of the predicted class (max_prob).

        Returns:
            Single input: (intent_name: str, bias_vector: dict, confidence: float).
            Batch input:  (intent_names: list[str], bias_vectors: list[dict], confidences: list[float]).
        """
        probs, single = self._forward(state_sequence, static_profile)
        if single:
            p = probs[0]
            pred_id = int(p.argmax())
            confidence = float(p[pred_id])
            name = INTENT_ID2NAME[pred_id]
            return name, INTENT_BIAS[name].copy(), confidence
        results = []
        for p in probs:
            pred_id = int(p.argmax())
            confidence = float(p[pred_id])
            name = INTENT_ID2NAME[pred_id]
            results.append((name, INTENT_BIAS[name].copy(), confidence))
        names       = [r[0] for r in results]
        bias_vecs   = [r[1] for r in results]
        confidences = [r[2] for r in results]
        return names, bias_vecs, confidences
