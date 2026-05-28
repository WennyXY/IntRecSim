import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from model.intent_models import build_intent_model, TransitionDataset, compute_loss

INTENT_NAMES = ["stay_profile", "follow_recent", "explore"]


def choose_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_split(data_dir, split):
    return np.load(Path(data_dir) / f"{split}.npz")


def build_loaders(data_dir, batch_size, num_workers, emb_dir=None, decay_factor=1.0):
    train_dataset = TransitionDataset(
        load_split(data_dir, "train"),
        emb_dir=emb_dir, decay_factor=decay_factor,
    )
    val_dataset = TransitionDataset(
        load_split(data_dir, "intent_eval"),
        emb_dir=emb_dir, decay_factor=decay_factor,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def infer_dims(data_dir, emb_dir=None, input_repr="auto"):
    train_data = load_split(data_dir, "train")
    dims = {
        "profile_dim": train_data["static_profile"].shape[1],
        "state_dim":   train_data["state_sequence"].shape[2],
        "num_intents": int(train_data["intent_label"].max()) + 1,
    }
    if emb_dir is not None and input_repr != "genre":
        items   = np.load(Path(emb_dir) / "items.npy")
        emb_dim = items.shape[1]
        if input_repr == "concat":
            # state: genre_dim + emb_dim;  profile: genre_dim + emb_dim
            dims["state_dim"]   = dims["state_dim"]   + emb_dim
            dims["profile_dim"] = dims["profile_dim"] + emb_dim
        else:  # "embedding" or "auto"
            dims["state_dim"]   = emb_dim
            dims["profile_dim"] = emb_dim
    return dims


def compute_class_weights(data_dir, num_intents, device, mode="inv"):
    """Class weights normalised to mean=1.

    mode:
      "inv"  — 1/count  (default, stronger correction for severe imbalance)
      "sqrt" — 1/sqrt(count) (softer, avoids over-correcting with very few samples)
    """
    labels = load_split(data_dir, "train")["intent_label"]
    counts = np.bincount(labels, minlength=num_intents).astype(np.float32)
    counts = np.maximum(counts, 1)
    weights = 1.0 / counts if mode == "inv" else 1.0 / np.sqrt(counts)
    weights = weights / weights.mean()
    dist = {INTENT_NAMES[i]: f"{counts[i]/counts.sum():.1%}" for i in range(num_intents)}
    print(f"train label dist : {dist}")
    print(f"class_weights ({mode}): {dict(zip(range(num_intents), weights.round(3)))}")
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _get_inputs(batch, device, input_repr="auto"):
    """Return (state_seq, profile) from a batch.

    input_repr choices:
      "auto"      — embedding tensors when present, genre histograms otherwise.
      "genre"     — always genre histograms regardless of embedding availability.
      "embedding" — L2-normalised semantic embeddings only (requires emb_dir).
      "concat"    — cat([genre, embedding], dim=-1) for both state and profile.
    """
    has_emb = "recent_state_emb" in batch
    if input_repr == "concat" and has_emb:
        # L2-normalize the genre slice (all dims except the trailing drift scalar)
        # so genre and embedding halves each have unit norm — matching how concat
        # similarity is computed in intent_data_construction_2.py. Drift is kept
        # raw since it is not part of the label similarity computation.
        genre_s = torch.nn.functional.normalize(batch["state_seq"][..., :-1].float(), dim=-1)
        drift_s = batch["state_seq"][..., -1:].float()
        state   = torch.cat([genre_s, drift_s, batch["recent_state_emb"]], dim=-1).to(device)
        genre_p = torch.nn.functional.normalize(batch["profile"].float(), dim=-1)
        profile = torch.cat([genre_p, batch["profile_emb"]], dim=-1).to(device)
        return state, profile
    if input_repr in ("embedding", "auto") and has_emb:
        return batch["recent_state_emb"].to(device), batch["profile_emb"].to(device)
    return batch["state_seq"].to(device), batch["profile"].to(device)


def train_one_epoch(model, loader, optimizer, device,
                    class_weights=None, label_smoothing=0.0, grad_clip=1.0,
                    input_repr="auto"):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        state_seq, profile = _get_inputs(batch, device, input_repr)
        intent = batch["intent"].to(device)

        optimizer.zero_grad()
        logits = model(state_seq, profile)
        loss = compute_loss(logits, intent, weight=class_weights, label_smoothing=label_smoothing)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        n = intent.size(0)
        total_loss += loss.item() * n
        correct    += (logits.argmax(dim=-1) == intent).sum().item()
        total      += n

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, num_intents, input_repr="auto"):
    """Unweighted loss, overall accuracy, macro-F1, per-class accuracy and F1."""
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
        total_loss += compute_loss(logits, intent).item() * intent.size(0)
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

    acc       = float(tp.sum() / total)
    precision = tp / np.maximum(tp + fp, 1)
    recall    = tp / np.maximum(tp + fn, 1)
    f1_per    = 2 * precision * recall / np.maximum(precision + recall, 1e-8)
    macro_f1  = float(f1_per.mean())
    per_acc   = (class_correct / np.maximum(class_total, 1)).tolist()

    return total_loss / total, acc, macro_f1, per_acc, f1_per.tolist()


def train_model(model, train_loader, val_loader, args, class_weights=None):
    device      = torch.device(args.device)
    num_intents = args.num_intents
    model.to(device)

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
        *[f"val_acc_{n}"  for n in intent_names],
        *[f"val_f1_{n}"   for n in intent_names],
        "best_macro_f1", "checkpoint_saved",
    ]
    with log_path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device,
            class_weights=class_weights,
            label_smoothing=args.label_smoothing,
            grad_clip=args.grad_clip,
            input_repr=args.input_repr,
        )
        val_loss, val_acc, val_macro_f1, per_acc, per_f1 = evaluate(
            model, val_loader, device, num_intents,
            input_repr=args.input_repr,
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
        "best_macro_f1":  best_macro_f1,
        "checkpoint_path": str(checkpoint_path),
        "log_path":        str(log_path),
    }
    with (args.output_dir / "training_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"best_macro_f1 = {best_macro_f1:.4f}  ckpt = {args.output_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config",        type=Path, default="config/intent_gru.yaml")
    parser.add_argument("-d","--data-dir",      type=Path, default=Path("output/state_transition/ml_1m/score_cosine_all_v2/"))
    parser.add_argument("--output-dir",    type=Path, default=Path("output/state_transition/"))
    parser.add_argument("--model-type",    default="gru", choices=["gru", "transformer", "tcn"])
    parser.add_argument("--epochs",        type=int,   default=50)
    parser.add_argument("--batch-size",    type=int,   default=512)
    parser.add_argument("--hidden-dim",    type=int,   default=128)
    parser.add_argument("--nhead",         type=int,   default=4)
    parser.add_argument("--max-seq-len",   type=int,   default=16)
    parser.add_argument("--num-layers",    type=int,   default=2)
    parser.add_argument("--kernel-size",   type=int,   default=3)
    parser.add_argument("--dropout",       type=float, default=0.0)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--weight-decay",  type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--grad-clip",     type=float, default=1.0, help="Max gradient norm (0 = disabled).")
    parser.add_argument("--patience",      type=int,   default=20, help="Early-stopping patience in epochs (0 = disabled).")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--class-weight-mode", default="inv", choices=["inv", "sqrt"],
                        help="Class weight formula: inv=1/count (default), sqrt=1/sqrt(count) (softer).")
    parser.add_argument("--num-workers",   type=int,   default=3)
    parser.add_argument("--device",        default=choose_device())
    parser.add_argument("--seed",          type=int,   default=2026)
    parser.add_argument("--emb-dir",       type=Path,  default=None,
                        help="Path to embeddings/ directory (items.npy, profiles.npy). "
                             "Defaults to {data-dir}/embeddings if that directory exists.")
    parser.add_argument("--decay-factor",  type=float, default=1.0,
                        help="Exponential decay for window item embeddings (1.0 = uniform).")
    parser.add_argument("--input-repr",    default="auto",
                        choices=["auto", "genre", "embedding", "concat"],
                        help="Model input representation. "
                             "auto: embedding when emb-dir set, genre otherwise. "
                             "genre: 18-dim genre histogram only. "
                             "embedding: semantic embedding only. "
                             "concat: cat(genre, embedding).")

    args, _ = parser.parse_known_args()
    if args.config is not None and args.config.exists():
        with args.config.open() as f:
            cfg = yaml.safe_load(f)
        parser.set_defaults(**{k.replace("-", "_"): v for k, v in cfg.items() if k != "config"})

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = args.data_dir.parts
    if any("ml" in p for p in parts):
        dataset_name = "ml_1m"
    elif any("book" in p for p in parts):
        dataset_name = "book"
    else:
        dataset_name = "beauty"

    args.output_dir = args.output_dir / dataset_name / args.model_type / f"ckpt_{timestamp}"
    return args


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
        if 'hyb' in str(args.data_dir):
            args.input_repr = "concat"
        elif 'emb' in str(args.data_dir):
            args.input_repr = "embedding"
        else:
            args.input_repr = "genre"
    # Validate that the data contains embedding index arrays when needed.
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

    train_loader, val_loader = build_loaders(
        args.data_dir, args.batch_size, args.num_workers,
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

    device = torch.device(args.device)
    class_weights = None
    if not args.no_class_weights:
        class_weights = compute_class_weights(
            args.data_dir, dims["num_intents"], device, mode=args.class_weight_mode
        )
    print(f"model={args.model_type}  device={args.device}  dims={dims}")
    print(f"label_smoothing={args.label_smoothing}  grad_clip={args.grad_clip}  patience={args.patience}")
    print(f"class_weights={'on' if class_weights is not None else 'off'}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "train_config.json").open("w") as f:
        json.dump({**vars(args), **dims, "mode": mode}, f, indent=2, default=str)

    train_model(model, train_loader, val_loader, args, class_weights=class_weights)


if __name__ == "__main__":
    main()
