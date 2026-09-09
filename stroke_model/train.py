"""학습 루프. 01_Training.ipynb의 학습 부분과 동일 (변경 없음)."""
import time
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import StrokeDataset, collate_fn, move_batch, maybe_apply_mixup
from .losses import total_loss, compute_metrics
from .model import SwinWarpHintStrokeModel
from .utils import (ensure_dir, get_device, set_seed, safe_json_dump,
                     save_checkpoint, average_logs)


def build_optimizer(model: SwinWarpHintStrokeModel, config: Dict[str, Any]) -> torch.optim.Optimizer:
    backbone_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone_params if name.startswith("swin.") else other_params).append(p)
    return torch.optim.AdamW(
        [{"params": other_params, "lr": config["lr"]},
         {"params": backbone_params, "lr": config["backbone_lr"]}],
        weight_decay=config["weight_decay"],
    )


def make_loaders(config: Dict[str, Any]):
    train_ds = StrokeDataset(config["train_csv_path"], config["dataset_root"], train=True, config=config)
    test_ds = StrokeDataset(config["test_csv_path"], config["test_dataset_root"], train=False, config=config)
    common = dict(batch_size=config["batch_size"], num_workers=config["num_workers"],
                  pin_memory=torch.cuda.is_available(), collate_fn=collate_fn)
    return (DataLoader(train_ds, shuffle=True, **common),
            DataLoader(test_ds, shuffle=False, **common))


def make_training_objects(config: Dict[str, Any], device):
    model = SwinWarpHintStrokeModel(config).to(device)
    if config["freeze_backbone_epochs"] > 0:
        model.set_backbone_trainable(False)
    optimizer = build_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config["epochs"]), eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(config["amp"] and device.type == "cuda"))
    return model, optimizer, scheduler, scaler


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, config) -> Dict[str, float]:
    model.train()
    logs = []
    pbar = tqdm(loader, desc=f"train {epoch}", leave=False)
    for batch in pbar:
        batch = move_batch(batch, device)
        batch = maybe_apply_mixup(batch, config)

        optimizer.zero_grad(set_to_none=True)
        use_amp = config["amp"] and device.type == "cuda"
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(batch["I_prep"], batch["I_g"], batch["I_G_strokes"])
            loss, log = total_loss(outputs, batch["target_strokes"], epoch, config)

        if use_amp:
            scaler.scale(loss).backward()
            if config["grad_clip"] is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if config["grad_clip"] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()

        with torch.no_grad():
            log.update(compute_metrics(outputs["logits"], batch["target_strokes"], config))
        logs.append(log)
        pbar.set_postfix({"loss": f"{log['loss']:.3f}", "sdice": f"{log['soft_dice']:.3f}"})
    return average_logs(logs)


@torch.no_grad()
def evaluate(model, loader, device, epoch, config) -> Dict[str, float]:
    model.eval()
    logs = []
    pbar = tqdm(loader, desc=f"test {epoch}", leave=False)
    for batch in pbar:
        batch = move_batch(batch, device)
        outputs = model(batch["I_prep"], batch["I_g"], batch["I_G_strokes"])
        loss, log = total_loss(outputs, batch["target_strokes"], epoch, config)
        log.update(compute_metrics(outputs["logits"], batch["target_strokes"], config))
        logs.append(log)
        pbar.set_postfix({"loss": f"{log['loss']:.3f}", "sdice": f"{log['soft_dice']:.3f}"})
    return average_logs(logs)


def save_epoch_checkpoints(ckpt_dir, model, optimizer, scheduler, epoch, best_score,
                            history, test_log, config):
    score = test_log["soft_dice"] if test_log is not None else -1.0
    if test_log is not None and score > best_score:
        best_score = score
        save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scheduler, epoch, best_score, history, config)
        print(f"  saved best.pt: {best_score:.4f}")
    save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler, epoch, best_score, history, config)
    if epoch % config["save_every"] == 0:
        save_checkpoint(ckpt_dir / f"epoch_{epoch:04d}.pt", model, optimizer, scheduler, epoch, best_score, history, config)
    return best_score


def run_training(config: Dict[str, Any]):
    set_seed(config["seed"])
    device = get_device()
    out_dir = ensure_dir(config["out_dir"])
    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    safe_json_dump(config, out_dir / "config.json")

    print("device:", device)
    print("train :", config["train_csv_path"])
    print("test  :", config["test_csv_path"])
    print("output:", out_dir)

    train_loader, test_loader = make_loaders(config)
    model, optimizer, scheduler, scaler = make_training_objects(config, device)

    history, best_score = [], -1.0

    for epoch in range(1, config["epochs"] + 1):
        t0 = time.time()

        if epoch == config["freeze_backbone_epochs"] + 1 and config["freeze_backbone_epochs"] > 0:
            print("[unfreeze] Swin backbone")
            model.set_backbone_trainable(True)
            optimizer = build_optimizer(model, config)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, config["epochs"] - epoch + 1), eta_min=1e-6)

        train_log = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, config)

        do_eval = (epoch % config["eval_every"] == 0) or (epoch == config["epochs"])
        test_log = evaluate(model, test_loader, device, epoch, config) if do_eval else None
        scheduler.step()

        row = {"epoch": epoch, "elapsed_sec": time.time() - t0, "train": train_log, "test": test_log}
        history.append(row)
        safe_json_dump(history, out_dir / "history.json")

        msg = f"[{epoch:03d}] train loss={train_log['loss']:.4f}, dice={train_log['soft_dice']:.4f}"
        if test_log is not None:
            msg += f" | test loss={test_log['loss']:.4f}, dice={test_log['soft_dice']:.4f}"
        print(msg)

        best_score = save_epoch_checkpoints(ckpt_dir, model, optimizer, scheduler, epoch,
                                             best_score, history, test_log, config)

    print("\nDone.")
    print("best:", ckpt_dir / "best.pt")
    print("last:", ckpt_dir / "last.pt")
    return model, history
