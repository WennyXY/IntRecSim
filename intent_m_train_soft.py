import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset

from model.intent_models import build_intent_model, TransitionDataset

INTENT_NAMES = ["stay_profile", "follow_recent", "explore"]
SCORE_KEYS   = ["score_stay_profile", "score_follow_recent", "score_explore"]


# ---------------------------------------------------------------------------
# Soft-label dataset
# ---------------------------------------------------------------------------

class SoftLabelTransitionDataset(Dataset):
    """Loads evidence scores from the NPZ file and converts to soft labels.

    soft_label[i] = score_vec[i] ** (1/temperature) / Z   (renormalised)

    temperature=1.0  → proportional to raw scores (default)
    temperature<1.0  → sharper, concentrates mass on the argmax class
    temperature>1.0  → softer, approaches uniform distribution

    When ``emb_dir`` is provided and the npz contains ``window_item_ids``,
    embedding lookup tables are loaded and per-sample embeddings are
    reconstructed on-the-fly in __getitem__ — identical to TransitionDataset.
    """

    def __init__(self, npz_data, temperature: float = 1.0,
                 emb_dir=None, decay_factor: float = 1.0):
        self.static_profile = torch.as_tensor(npz_data["static_profile"], dtype=torch.float32)
        self.state_sequence = torch.as_tensor(npz_data["state_sequence"], dtype=torch.float32)
        self.intent_label   = torch.as_tensor(npz_data["intent_label"],   dtype=torch.long)

        # stack raw scores → (N, 3)
        raw = np.stack([npz_data[k] for k in SCORE_KEYS], axis=1).astype(np.float64)

        # temperature scaling then L1 normalise
        if temperature != 1.0:
            raw = raw ** (1.0 / temperature)
        totals = raw.sum(axis=1, keepdims=True)
        totals = np.where(totals == 0, 1.0, totals)   # guard against all-zero rows
        soft = (raw / totals).astype(np.float32)

        self.soft_label = torch.as_tensor(soft, dtype=torch.float32)

        # Embedding lookup tables (mirrors TransitionDataset)
        self.use_emb = False
        if emb_dir is not None and "window_item_ids" in npz_data:
            emb_dir = Path(emb_dir)
            self.item_embs    = torch.as_tensor(
                np.load(emb_dir / "items.npy").astype(np.float32)
            )
            self.profile_embs = torch.as_tensor(
                np.load(emb_dir / "profiles.npy").astype(np.float32)
            )
            self.window_item_ids = torch.as_tensor(
                npz_data["window_item_ids"], dtype=torch.long
            )
            self.target_item_id  = torch.as_tensor(
                npz_data["target_item_id"],  dtype=torch.long
            )
            self.profile_emb_idx = torch.as_tensor(
                npz_data["profile_emb_idx"], dtype=torch.long
            )
            self._decay_w = self._make_decay_weights(
                decay_factor, self.window_item_ids.shape[-1]
            )
            self.use_emb = True

    @staticmethod
    def _make_decay_weights(decay_factor: float, window_size: int):
        if decay_factor == 1.0:
            return None
        w = torch.tensor(
            [decay_factor ** (window_size - 1 - i) for i in range(window_size)],
            dtype=torch.float32,
        )
        return w / w.sum()

    def __len__(self):
        return len(self.intent_label)

    def __getitem__(self, idx):
        item = {
            "state_seq":  self.state_sequence[idx],
            "profile":    self.static_profile[idx],
            "intent":     self.intent_label[idx],       # hard label (for eval only)
            "soft_label": self.soft_label[idx],         # soft distribution (for loss)
        }
        if self.use_emb:
            window_ids  = self.window_item_ids[idx]        # (T, W)
            window_embs = self.item_embs[window_ids]        # (T, W, D)
            if self._decay_w is None:
                recent_emb = window_embs.mean(dim=1)        # (T, D)
            else:
                recent_emb = (window_embs * self._decay_w[None, :, None]).sum(dim=1)
            item["profile_emb"]      = self.profile_embs[self.profile_emb_idx[idx]]
            item["recent_state_emb"] = F.normalize(recent_emb, dim=-1)
            item["target_emb"]       = self.item_embs[self.target_item_id[idx]]
        return item


# ---------------------------------------------------------------------------
# Input selection (mirrors intent_m_train._get_inputs exactly)
# ---------------------------------------------------------------------------

def _get_inputs(batch, device, input_repr="auto"):
    """Return (state_seq, profile) from a batch.

    input_repr choices:
      "auto"      — embedding tensors when present, genre histograms otherwise.
      "genre"     — always genre histograms regardless of embedding availability.
      "embedding" — L2-normalised semantic embeddings only (requires emb_dir).
      "concat"    — cat([L2_norm(genre), drift, embedding], dim=-1) for state;
                    cat([L2_norm(genre), embedding], dim=-1) for profile.
    """
    has_emb = "recent_state_emb" in batch
    if input_repr == "concat" and has_emb:
        # L2-normalize the genre profile slice (all dims except the trailing drift
        # scalar) so it has unit norm like the embedding — matching how concat
        # similarity is computed in intent_data_construction_2.py.
        genre_s = F.normalize(batch["state_seq"][..., :-1].float(), dim=-1)
        drift_s = batch["state_seq"][..., -1:].float()
        state   = torch.cat([genre_s, drift_s, batch["recent_state_emb"]], dim=-1).to(device)
        genre_p = F.normalize(batch["profile"].float(), dim=-1)
        profile = torch.cat([genre_p, batch["profile_emb"]], dim=-1).to(device)
        return state, profile
    if input_repr in ("embedding", "auto") and has_emb:
        return batch["recent_state_emb"].to(device), batch["profile_emb"].to(device)
    return batch["state_seq"].to(device), batch["profile"].to(device)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def soft_cross_entropy(logits, soft_targets):
    """Soft-target cross-entropy: -Σ p_i · log q_i  (averaged over batch).

    Equivalent to KL(soft_targets ‖ softmax(logits)) + H(soft_targets).
    No class weighting — soft labels already encode per-sample uncertainty.
    """
    log_probs = F.log_softmax(logits, dim=-1)          # (B, C)
    return -(soft_targets * log_probs).sum(dim=-1).mean()


def hybrid_cross_entropy(logits, soft_targets, hard_labels, class_weights, lam):
    """Hybrid loss: soft-CE for calibration + weighted hard-CE to recover minority recall.

    L = (1 - lam) * soft_CE(logits, soft_targets)
      +       lam  * weighted_CE(logits, hard_labels, weight=class_weights)

    lam=0.0  → pure soft-CE (same as soft_cross_entropy above)
    lam=1.0  → pure weighted hard-CE (same as original intent_m_train.py)
    lam=0.3-0.5 → recommended range: calibration from soft labels + minority-class
                  recall recovery from class weights.
    """
    soft = soft_cross_entropy(logits, soft_targets)
    hard = F.cross_entropy(logits, hard_labels, weight=class_weights)
    return (1.0 - lam) * soft + lam * hard


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_split(data_dir, split):
    return np.load(Path(data_dir) / f"{split}.npz")


def build_loaders(data_dir, batch_size, num_workers, temperature,
                  emb_dir=None, decay_factor=1.0):
    train_npz = load_split(data_dir, "train")
    val_npz   = load_split(data_dir, "intent_eval")

    missing = [k for k in SCORE_KEYS if k not in train_npz.files]
    if missing:
        raise ValueError(
            f"NPZ is missing soft-label score fields: {missing}.\n"
            "Re-generate the dataset with intent_data_construction_2.py "
            "(requires intent_method='score' or any threshold variant that saves scores)."
        )

    train_dataset = SoftLabelTransitionDataset(
        train_npz, temperature=temperature,
        emb_dir=emb_dir, decay_factor=decay_factor,
    )
    val_dataset = TransitionDataset(
        val_npz, emb_dir=emb_dir, decay_factor=decay_factor,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def infer_dims(data_dir, emb_dir=None, input_repr="auto"):
    d = load_split(data_dir, "train")
    dims = {
        "profile_dim": d["static_profile"].shape[1],
        "state_dim":   d["state_sequence"].shape[2],
        "num_intents": int(d["intent_label"].max()) + 1,
    }
    if emb_dir is not None and input_repr != "genre":
        items   = np.load(Path(emb_dir) / "items.npy")
        emb_dim = items.shape[1]
        if input_repr == "concat":
            dims["state_dim"]   = dims["state_dim"]   + emb_dim
            dims["profile_dim"] = dims["profile_dim"] + emb_dim
        else:  # "embedding" or "auto"
            dims["state_dim"]   = emb_dim
            dims["profile_dim"] = emb_dim
    return dims


def print_label_stats(data_dir, num_intents, temperature):
    d = load_split(data_dir, "train")
    labels = d["intent_label"]
    counts = np.bincount(labels, minlength=num_intents)
    dist   = {INTENT_NAMES[i]: f"{counts[i]/counts.sum():.1%}" for i in range(num_intents)}
    print(f"train hard label dist : {dist}")
    raw = np.stack([d[k] for k in SCORE_KEYS], axis=1).astype(np.float64)
    if temperature != 1.0:
        raw = raw ** (1.0 / temperature)
    totals = raw.sum(axis=1, keepdims=True)
    soft = raw / np.where(totals == 0, 1.0, totals)
    mean_soft = soft.mean(axis=0)
    mean_soft_str = {INTENT_NAMES[i]: f"{mean_soft[i]:.3f}" for i in range(num_intents)}
    print(f"mean soft label (T={temperature}) : {mean_soft_str}")


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, grad_clip=1.0, input_repr="auto",
                    class_weights=None, hybrid_lambda=0.0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        state_seq, profile = _get_inputs(batch, device, input_repr)
        hard_intent = batch["intent"].to(device)
        soft_target = batch["soft_label"].to(device)

        optimizer.zero_grad()
        logits = model(state_seq, profile)
        if hybrid_lambda > 0.0 and class_weights is not None:
            loss = hybrid_cross_entropy(logits, soft_target, hard_intent,
                                        class_weights, hybrid_lambda)
        else:
            loss = soft_cross_entropy(logits, soft_target)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        n = hard_intent.size(0)
        total_loss += loss.item() * n
        correct    += (logits.argmax(dim=-1) == hard_intent).sum().item()
        total      += n

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, num_intents, input_repr="auto"):
    """Unweighted hard-label evaluation: loss, accuracy, macro-F1, per-class metrics."""
    model.eval()
    total_loss, total = 0.0, 0
    tp = np.zeros(num_intents, dtype=np.int64)
    fp = np.zeros(num_intents, dtype=np.int64)
    fn = np.zeros(num_intents, dtype=np.int64)
    class_correct = np.zeros(num_intents, dtype=np.int64)
    class_total   = np.zeros(num_intents, dtype=np.int64)

    for batch in loader:
        state_seq, profile = _get_inputs(batch, device, input_repr)
        intent = batch["intent"].to(device)

        logits = model(state_seq, profile)
        total_loss += F.cross_entropy(logits, intent).item() * intent.size(0)
        total      += intent.size(0)

        preds  = logits.argmax(dim=-1).cpu().numpy()
        labels = intent.cpu().numpy()
        for c in range(num_intents):
            pred_c  = preds  == c
            label_c = labels == c
            tp[c] += int(( pred_c &  label_c).sum())
            fp[c] += int(( pred_c & ~label_c).sum())
            fn[c] += int((~pred_c &  label_c).sum())
            class_correct[c] += int(( pred_c &  label_c).sum())
            class_total[c]   += int(label_c.sum())

    acc      = float(tp.sum() / total)
    prec     = tp / np.maximum(tp + fp, 1)
    rec      = tp / np.maximum(tp + fn, 1)
    f1_per   = 2 * prec * rec / np.maximum(prec + rec, 1e-8)
    macro_f1 = float(f1_per.mean())
    per_acc  = (class_correct / np.maximum(class_total, 1)).tolist()

    return total_loss / total, acc, macro_f1, per_acc, f1_per.tolist()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _compute_class_weights(data_dir, num_intents, device):
    """sqrt inverse-frequency weights, mean-normalised (same formula as intent_m_train.py)."""
    labels  = np.load(Path(data_dir) / "train.npz")["intent_label"]
    counts  = np.bincount(labels, minlength=num_intents).astype(np.float32)
    counts  = np.maximum(counts, 1)
    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.mean()
    print(f"hybrid class_weights (sqrt): {dict(zip(range(num_intents), weights.round(3)))}")
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_model(model, train_loader, val_loader, args):
    device      = torch.device(args.device)
    num_intents = args.num_intents
    model.to(device)

    class_weights = None
    if args.hybrid_lambda > 0.0:
        class_weights = _compute_class_weights(args.data_dir, num_intents, device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )

    best_macro_f1 = 0.0
    no_improve    = 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_intent_model.pt"
    log_path        = args.output_dir / "training_log.csv"

    intent_names = INTENT_NAMES[:num_intents]
    fieldnames = [
        "epoch", "lr",
        "train_loss", "train_acc",
        "val_loss", "val_acc", "val_macro_f1",
        *[f"val_acc_{n}" for n in intent_names],
        *[f"val_f1_{n}"  for n in intent_names],
        "best_macro_f1", "checkpoint_saved",
    ]
    with log_path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device,
            grad_clip=args.grad_clip, input_repr=args.input_repr,
            class_weights=class_weights, hybrid_lambda=args.hybrid_lambda,
        )
        val_loss, val_acc, val_macro_f1, per_acc, per_f1 = evaluate(
            model, val_loader, device, num_intents, input_repr=args.input_repr,
        )
        scheduler.step(val_macro_f1)

        checkpoint_saved = val_macro_f1 > best_macro_f1
        if checkpoint_saved:
            best_macro_f1 = val_macro_f1
            no_improve    = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch":            epoch,
                "val_acc":          val_acc,
                "val_macro_f1":     val_macro_f1,
                "args":             vars(args),
            }, checkpoint_path)
        else:
            no_improve += 1

        current_lr = optimizer.param_groups[0]["lr"]
        log_row = {
            "epoch": epoch, "lr": current_lr,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "val_macro_f1": val_macro_f1,
            **{f"val_acc_{n}": per_acc[i] for i, n in enumerate(intent_names)},
            **{f"val_f1_{n}":  per_f1[i]  for i, n in enumerate(intent_names)},
            "best_macro_f1": best_macro_f1, "checkpoint_saved": checkpoint_saved,
        }
        with log_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(log_row)

        per_str = "  ".join(
            f"{n[:4]}={per_acc[i]:.3f}(f1={per_f1[i]:.3f})"
            for i, n in enumerate(intent_names)
        )
        print(
            f"epoch={epoch:03d} lr={current_lr:.1e} "
            f"train={train_loss:.4f}/{train_acc:.4f} "
            f"val={val_loss:.4f}/{val_acc:.4f} macro_f1={val_macro_f1:.4f}  {per_str}"
        )

        if args.patience > 0 and no_improve >= args.patience:
            print(f"Early stopping: no macro-F1 improvement for {args.patience} epochs.")
            break

    summary = {
        "best_macro_f1":    best_macro_f1,
        "soft_temperature": args.soft_temp,
        "hybrid_lambda":    args.hybrid_lambda,
        "checkpoint_path":  str(checkpoint_path),
        "log_path":         str(log_path),
    }
    with (args.output_dir / "training_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"best_macro_f1={best_macro_f1:.4f}  ckpt: {args.output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def choose_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Intent model training with soft labels derived from evidence scores."
    )
    parser.add_argument("--config",       type=Path, default=None,
                        help="Optional YAML config; CLI args override.")
    parser.add_argument("--data-dir",     type=Path,
                        default=Path("output/state_transition/ml_1m/processed_data/"))
    parser.add_argument("--output-dir",   type=Path,
                        default=Path("output/state_transition/"))
    parser.add_argument("--model-type",   default="gru", choices=["gru", "transformer", "tcn"])
    parser.add_argument("--soft-temp",      type=float, default=1.0,
                        help="Temperature for soft-label sharpening. "
                             "<1 sharpens toward argmax, >1 softens toward uniform.")
    parser.add_argument("--hybrid-lambda", type=float, default=0.0,
                        help="Mix weight for weighted hard-CE component (0=pure soft, 1=pure hard). "
                             "When >0, adds lam*weighted_CE(hard_labels) to recover minority recall. "
                             "Recommended range: 0.3-0.5 for class-imbalanced datasets.")
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch-size",   type=int,   default=512)
    parser.add_argument("--hidden-dim",   type=int,   default=128)
    parser.add_argument("--nhead",        type=int,   default=4)
    parser.add_argument("--max-seq-len",  type=int,   default=16)
    parser.add_argument("--num-layers",   type=int,   default=2)
    parser.add_argument("--kernel-size",  type=int,   default=3)
    parser.add_argument("--dropout",      type=float, default=0.0)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip",    type=float, default=1.0,
                        help="Gradient clipping max norm (0 = disabled).")
    parser.add_argument("--patience",     type=int,   default=20,
                        help="Early-stopping patience in epochs (0 = disabled).")
    parser.add_argument("--num-workers",  type=int,   default=0)
    parser.add_argument("--device",       default=choose_device())
    parser.add_argument("--seed",         type=int,   default=2026)
    parser.add_argument("--emb-dir",      type=Path,  default=None,
                        help="Path to embeddings/ directory (items.npy, profiles.npy). "
                             "Required for --input-repr embedding or concat.")
    parser.add_argument("--decay-factor", type=float, default=1.0,
                        help="Exponential decay for window item embeddings (1.0 = uniform).")
    parser.add_argument("--input-repr",   default="auto",
                        choices=["auto", "genre", "embedding", "concat"],
                        help="Model input representation: auto / genre / embedding / concat.")

    args, _ = parser.parse_known_args()
    if args.config is not None and args.config.exists():
        with args.config.open() as f:
            cfg = yaml.safe_load(f)
        parser.set_defaults(**{k.replace("-", "_"): v for k, v in cfg.items() if k != "config"})
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    parts = args.data_dir.parts
    if any("ml_1m" in p or "ml1m" in p for p in parts):
        dataset_name = "ml_1m"
    elif any("book" in p for p in parts):
        dataset_name = "book"
    else:
        dataset_name = "beauty"

    args.output_dir = (
        args.output_dir / dataset_name / f"{args.model_type}_soft" / f"ckpt_{timestamp}"
    )
    return args


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    emb_dir = args.emb_dir
    if emb_dir is None:
        candidate = args.data_dir / "embeddings"
        if (candidate / "items.npy").exists():
            emb_dir = candidate
            print(f"Auto-detected emb-dir: {emb_dir}")

    if args.input_repr == "auto":
        if "hyb" in str(args.data_dir):
            args.input_repr = "concat"
        elif "emb" in str(args.data_dir):
            args.input_repr = "embedding"
        else:
            args.input_repr = "genre"

    if args.input_repr in ("concat", "embedding") and emb_dir is not None:
        probe = load_split(args.data_dir, "train")
        if "window_item_ids" not in probe:
            raise ValueError(
                f"--input-repr {args.input_repr!r} requires 'window_item_ids' in the "
                f"training npz, but {args.data_dir / 'train.npz'} does not contain it.\n"
                f"Use --data-dir pointing to data built with an embedding_model "
                f"(e.g. score_cosine_all_v2_emb/), not the genre-only data."
            )

    dims = infer_dims(args.data_dir, emb_dir, input_repr=args.input_repr)
    args.num_intents = dims["num_intents"]

    if emb_dir is None or args.input_repr == "genre":
        mode = "genre"
    elif args.input_repr == "concat":
        mode = f"concat(genre+emb)  emb_dir={emb_dir}"
    else:
        mode = f"embedding  emb_dir={emb_dir}"
    print(f"input_repr={args.input_repr}  mode={mode}  state_dim={dims['state_dim']}  profile_dim={dims['profile_dim']}")

    print_label_stats(args.data_dir, dims["num_intents"], args.soft_temp)
    train_loader, val_loader = build_loaders(
        args.data_dir, args.batch_size, args.num_workers, args.soft_temp,
        emb_dir=emb_dir, decay_factor=args.decay_factor,
    )

    model = build_intent_model(
        model_type=args.model_type,
        state_dim=dims["state_dim"],
        profile_dim=dims["profile_dim"],
        num_intents=dims["num_intents"],
        hidden_dim=args.hidden_dim,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        kernel_size=args.kernel_size,
        max_seq_len=args.max_seq_len,
    )

    print(f"model={args.model_type}_soft  device={args.device}  dims={dims}")
    print(f"soft_temp={args.soft_temp}  grad_clip={args.grad_clip}  patience={args.patience}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "train_config.json").open("w") as f:
        json.dump({**vars(args), **dims, "mode": mode}, f, indent=2, default=str)

    train_model(model, train_loader, val_loader, args)


if __name__ == "__main__":
    main()
