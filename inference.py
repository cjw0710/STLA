import os
import time
import logging
from typing import Optional

import torch
from torch.utils.data import ConcatDataset, DataLoader, SequentialSampler

from config import parse_args
from utils.Setup import setup
from dataLoader import create_dataloaders
from graph import get_info
from model import DeDiff
from baseline import SimpleBaseline
from main import inference as eval_metrics


def resolve_checkpoint_path(args) -> Optional[str]:
    # Prefer explicit ckpt_file if it exists, else use saved_model_path/<dataset>.bin
    if args.ckpt_file and os.path.exists(args.ckpt_file):
        return args.ckpt_file
    candidate = os.path.join(args.saved_model_path, f"{args.dataset}.bin")
    if os.path.exists(candidate):
        return candidate
    return None


def load_weights(model: DeDiff, ckpt_path: str, device: str) -> None:
    state = torch.load(ckpt_path, map_location=device)
    key = 'model_state_dict' if isinstance(state, dict) and 'model_state_dict' in state else None
    to_load = state[key] if key else state
    model.load_state_dict(to_load, strict=True)


@torch.no_grad()
def measure_speed(args, model: DeDiff, dataloader, info, warmup_batches: int = 5):
    model.eval()
    total_batches = 0
    total_samples = 0
    start = None
    peak_alloc = 0
    peak_reserved = 0

    # Warmup (do not time)
    for i, batch in enumerate(dataloader):
        if i >= warmup_batches:
            break
        _ = model(args, batch, info)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Timed run over the full dataloader
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    for i, batch in enumerate(dataloader):
        if args.max_speed_batches and i >= args.max_speed_batches:
            break
        bs = batch['cascade'].size(0)
        _ = model(args, batch, info)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_batches += 1
        total_samples += bs
    elapsed = time.time() - start if start is not None else 0.0

    # Aggregate
    batches_per_sec = total_batches / elapsed if elapsed > 0 else 0.0
    samples_per_sec = total_samples / elapsed if elapsed > 0 else 0.0
    if torch.cuda.is_available():
        peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)

    return {
        'batches': total_batches,
        'samples': total_samples,
        'elapsed_sec': elapsed,
        'batches_per_sec': batches_per_sec,
        'samples_per_sec': samples_per_sec,
        'peak_mem_alloc_mb': peak_alloc,
        'peak_mem_reserved_mb': peak_reserved,
    }


def main():
    args = parse_args()
    setup(args)
    logging.info("Inference parameters: %s", args)

    # Data
    train_loader, val_loader, test_loader, train_dataset = create_dataloaders(args)
    info = get_info(args, train_loader, train_dataset)

    # Model
    if args.model == 'baseline':
        logging.info("Using SimpleBaseline model for inference")
        model = SimpleBaseline(args).to(args.device)
    else:
        model = DeDiff(args).to(args.device)
    ckpt_path = resolve_checkpoint_path(args)
    if ckpt_path is None:
        logging.warning(
            "No checkpoint found. Expected one of: %s or %s",
            args.ckpt_file,
            os.path.join(args.saved_model_path, f"{args.dataset}.bin"),
        )
    else:
        logging.info("Loading checkpoint: %s", ckpt_path)
        # Only load checkpoint for DeDiff (baseline may not have a trained checkpoint)
        if args.model == 'dediff':
            load_weights(model, ckpt_path, args.device)
        else:
            logging.info("Skip loading weights for baseline (no compatible checkpoint expected)")

    # Metrics on original test set (no repetition)
    metrics = eval_metrics(args, model, test_loader, info)
    print('# ---------- Test Metrics ----------')
    for k, v in metrics.items():
        print(f'{k}: {v:.6f}')

    # Speed test; optionally repeat test dataset to control batch count
    speed_loader = test_loader
    if args.repeat_test and args.repeat_test > 1:
        # Rebuild a repeated dataset-backed loader to increase number of batches while keeping samples identical
        base_ds = test_loader.dataset
        rep_ds = ConcatDataset([base_ds] * args.repeat_test)
        speed_loader = DataLoader(
            rep_ds,
            batch_size=args.batch_size,
            sampler=SequentialSampler(rep_ds),
            pin_memory=True,
        )

    print('\n# ---------- Inference Speed ----------')
    speed = measure_speed(args, model, speed_loader, info, warmup_batches=args.warmup_batches)
    print(f"batches: {speed['batches']}")
    print(f"samples: {speed['samples']}")
    print(f"elapsed_sec: {speed['elapsed_sec']:.4f}")
    print(f"batches_per_sec: {speed['batches_per_sec']:.4f}")
    print(f"samples_per_sec: {speed['samples_per_sec']:.4f}")
    if torch.cuda.is_available():
        print(f"peak_mem_alloc_mb: {speed['peak_mem_alloc_mb']:.2f}")
        print(f"peak_mem_reserved_mb: {speed['peak_mem_reserved_mb']:.2f}")


if __name__ == '__main__':
    main()
