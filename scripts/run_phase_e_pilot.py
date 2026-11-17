"""
RoomAI — Phase E Pilot: Decoder-Only SegFormer-B1 Indoor Adaptation
Trains only the MLP decode head (~0.57M params) on 48 indoor images (1% split).
Monitors on Validation Tuning Set (50 images).
Evaluates zero-shot vs fine-tuned on Quarantined Golden Test Set (200 images).
Saves checkpoint to checkpoints/experimental/segformer_b1_pilot/
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
from PIL import Image

# Suppress TensorFlow probing
os.environ["TRANSFORMERS_NO_TF"] = "1"
sys.modules["tensorflow"] = None

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

REPO_ROOT = Path(__file__).resolve().parent.parent
ADE_ROOT = REPO_ROOT / "archive (2)" / "ADEChallengeData2016"
SPLITS_DIR = REPO_ROOT / "data" / "splits"
REPORTS_DIR = REPO_ROOT / "reports"
CKPT_DIR = REPO_ROOT / "checkpoints" / "experimental" / "segformer_b1_pilot"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_CLASSES_10 = {
    1: "wall",
    4: "floor",
    6: "ceiling",
    9: "windowpane",
    15: "door",
    8: "bed",
    24: "sofa",
    20: "chair",
    16: "table",
    11: "cabinet",
}


class ADEIndoorDataset(Dataset):
    def __init__(self, manifest_path, split_folder, processor):
        with open(manifest_path, "r") as f:
            self.image_ids = json.load(f)
        self.split_folder = split_folder
        self.processor = processor

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = ADE_ROOT / "images" / self.split_folder / f"{img_id}.jpg"
        mask_path = ADE_ROOT / "annotations" / self.split_folder / f"{img_id}.png"

        img = Image.open(img_path).convert("RGB")
        mask = np.array(Image.open(mask_path))

        # ADE mask: 0 is unlabeled -> map to 255 for PyTorch CrossEntropy ignore_index
        # 1..150 map to 0..149 to match SegFormer 0-indexed logits
        label = np.full_like(mask, 255, dtype=np.int64)
        valid = (mask >= 1) & (mask <= 150)
        label[valid] = mask[valid] - 1

        encoded = self.processor(images=img, return_tensors="pt")
        pixel_values = encoded.pixel_values[0]

        # Resize label to 512x512 with nearest neighbor for training
        label_pil = Image.fromarray(label.astype(np.uint8)).resize((512, 512), Image.NEAREST)
        label_tensor = torch.from_numpy(np.array(label_pil, dtype=np.int64))

        return {
            "pixel_values": pixel_values,
            "labels": label_tensor,
            "img_id": img_id,
            "orig_size": (mask.shape[1], mask.shape[0]),  # (W, H)
            "orig_mask": mask,
        }


def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {
        "pixel_values": pixel_values,
        "labels": labels,
        "raw_items": batch,
    }


def evaluate_dataset(model, processor, dataset, desc="Evaluation"):
    model.eval()
    num_classes = 150
    conf_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=255)

    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    with torch.no_grad():
        for batch in loader:
            pv = batch["pixel_values"].to(DEVICE)
            lbl = batch["labels"].to(DEVICE)
            with torch.cuda.amp.autocast():
                out = model(pixel_values=pv, labels=lbl)
                loss = out.loss
                total_loss += float(loss.item()) if loss is not None else 0.0

            logits = out.logits
            # Evaluate against original unresized masks (vectorized bincount)
            for i, raw_item in enumerate(batch["raw_items"]):
                orig_w, orig_h = raw_item["orig_size"]
                gt_mask = raw_item["orig_mask"]

                item_logits = logits[i : i + 1]
                upsampled = torch.nn.functional.interpolate(
                    item_logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False
                )
                pred_mask = upsampled.argmax(dim=1)[0].cpu().numpy() + 1  # 1-based ADE index

                valid = (gt_mask >= 1) & (gt_mask <= num_classes)
                gt_v = gt_mask[valid].astype(np.int64)
                pd_v = pred_mask[valid].astype(np.int64)
                category_idx = (num_classes + 1) * gt_v + pd_v
                bincount = np.bincount(category_idx, minlength=(num_classes + 1) ** 2)
                conf_matrix += bincount.reshape(num_classes + 1, num_classes + 1)

    mean_loss = total_loss / max(1, len(loader))

    # Calculate per-class IoU for 10 target classes
    target_ious = {}
    for cid, name in TARGET_CLASSES_10.items():
        tp = conf_matrix[cid, cid]
        fp = conf_matrix[:, cid].sum() - tp
        fn = conf_matrix[cid, :].sum() - tp
        denom = tp + fp + fn
        target_ious[name] = float(tp / denom) if denom > 0 else 0.0

    target_10_miou = float(np.mean(list(target_ious.values())))

    # Calculate mIoU across all present classes
    present_ious = []
    for c in range(1, num_classes + 1):
        tp = conf_matrix[c, c]
        fp = conf_matrix[:, c].sum() - tp
        fn = conf_matrix[c, :].sum() - tp
        denom = tp + fp + fn
        if denom > 0:
            present_ious.append(tp / denom)
    all_present_miou = float(np.mean(present_ious)) if present_ious else 0.0

    return {
        "mean_loss": round(mean_loss, 4),
        "target_10_miou": round(target_10_miou, 4),
        "all_present_miou": round(all_present_miou, 4),
        "per_class_iou": {k: round(v, 4) for k, v in target_ious.items()},
    }


def main():
    print("===============================================================")
    print("RoomAI Phase E Pilot: Decoder-Only SegFormer-B1 Adaptation")
    print("===============================================================")

    model_id = "nvidia/segformer-b1-finetuned-ade-512-512"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_id).to(DEVICE)

    # Parameter accounting & freezing
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.segformer.parameters())
    decoder_params = sum(p.numel() for p in model.decode_head.parameters())

    # Freeze encoder
    for param in model.segformer.parameters():
        param.requires_grad = False
    # Ensure decode head is trainable
    for param in model.decode_head.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Model Parameters     : {total_params:,} ({total_params/1e6:.2f} M)")
    print(f"Frozen Encoder Parameters  : {encoder_params:,} ({encoder_params/1e6:.2f} M)")
    print(f"Trainable Decoder Parameters: {trainable_params:,} ({trainable_params/1e6:.2f} M, {trainable_params/total_params*100:.1f}%)")

    # Load datasets
    train_dataset = ADEIndoorDataset(SPLITS_DIR / "train_1pct_48.json", "training", processor)
    val_tuning_dataset = ADEIndoorDataset(SPLITS_DIR / "validation_tuning_50.json", "validation", processor)
    golden_test_dataset = ADEIndoorDataset(SPLITS_DIR / "golden_test_200.json", "validation", processor)

    print(f"\nLoaded Datasets:")
    print(f"  Training Split (1%)    : {len(train_dataset)} images")
    print(f"  Validation Tuning Set  : {len(val_tuning_dataset)} images")
    print(f"  Quarantined Golden Test: {len(golden_test_dataset)} images")

    # Pre-training Baseline Evaluation on Golden Test Set
    print("\n[Step 1] Evaluating Frozen Pretrained Baseline on Golden Test Set (200 images)...")
    baseline_metrics = evaluate_dataset(model, processor, golden_test_dataset, desc="Pretrained Baseline")
    print(f"  Baseline Target-10 mIoU : {baseline_metrics['target_10_miou']:.4f}")
    print(f"  Baseline All-Present mIoU: {baseline_metrics['all_present_miou']:.4f}")
    print("  Baseline Target-10 Per-Class IoU:")
    for k, v in baseline_metrics["per_class_iou"].items():
        print(f"    {k:12s}: {v:.4f}")

    # Training Loop
    print("\n[Step 2] Executing Pilot Training (3 Epochs, lr=1e-4, batch=4)...")
    optimizer = torch.optim.AdamW(model.decode_head.parameters(), lr=1e-4, weight_decay=0.01)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    scaler = torch.cuda.amp.GradScaler()

    torch.cuda.reset_peak_memory_stats()
    t_train_start = time.perf_counter()
    epoch_logs = []

    for epoch in range(1, 4):
        model.train()
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            pv = batch["pixel_values"].to(DEVICE)
            lbl = batch["labels"].to(DEVICE)

            with torch.cuda.amp.autocast():
                outputs = model(pixel_values=pv, labels=lbl)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.item())

        avg_loss = epoch_loss / len(train_loader)
        # Validation on Tuning Set ONLY (Golden Test Set is strictly untouched during training)
        val_eval = evaluate_dataset(model, processor, val_tuning_dataset, desc=f"Val Epoch {epoch}")
        epoch_logs.append({
            "epoch": epoch,
            "train_loss": round(avg_loss, 4),
            "val_loss": val_eval["mean_loss"],
            "val_target_10_miou": val_eval["target_10_miou"],
        })
        print(f"  Epoch {epoch}/3 | Train Loss: {avg_loss:.4f} | Val Loss: {val_eval['mean_loss']:.4f} | Val Target-10 mIoU: {val_eval['target_10_miou']:.4f}")

    train_time_sec = time.perf_counter() - t_train_start
    peak_train_vram_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_train_vram_res = torch.cuda.max_memory_reserved() / (1024 ** 2)

    print(f"\nPilot Training Completed in {train_time_sec:.1f}s ({train_time_sec/60:.2f} min).")
    print(f"Peak Training VRAM Allocated: {peak_train_vram_alloc:.1f} MB | Reserved: {peak_train_vram_res:.1f} MB")

    # Save fine-tuned pilot checkpoint
    model.save_pretrained(CKPT_DIR)
    processor.save_pretrained(CKPT_DIR)
    print(f"Pilot checkpoint successfully saved to: {CKPT_DIR}")

    # Post-training Final Evaluation on Quarantined Golden Test Set
    print("\n[Step 3] Evaluating Fine-Tuned Pilot on Quarantined Golden Test Set (200 images)...")
    post_metrics = evaluate_dataset(model, processor, golden_test_dataset, desc="Fine-Tuned Pilot")
    print(f"  Fine-Tuned Target-10 mIoU : {post_metrics['target_10_miou']:.4f}")
    print(f"  Fine-Tuned All-Present mIoU: {post_metrics['all_present_miou']:.4f}")
    print("  Fine-Tuned Target-10 Per-Class IoU:")
    for k, v in post_metrics["per_class_iou"].items():
        print(f"    {k:12s}: {v:.4f}")

    # Comparison & Delta Analysis
    delta_target_10 = post_metrics["target_10_miou"] - baseline_metrics["target_10_miou"]
    delta_planar = (
        (post_metrics["per_class_iou"]["wall"] + post_metrics["per_class_iou"]["floor"] + post_metrics["per_class_iou"]["ceiling"]) -
        (baseline_metrics["per_class_iou"]["wall"] + baseline_metrics["per_class_iou"]["floor"] + baseline_metrics["per_class_iou"]["ceiling"])
    ) / 3.0

    print("\n===============================================================")
    print("PILOT OUTCOME SUMMARY")
    print("===============================================================")
    print(f"Pretrained Baseline Target-10 mIoU : {baseline_metrics['target_10_miou']:.4f}")
    print(f"Fine-Tuned Pilot Target-10 mIoU    : {post_metrics['target_10_miou']:.4f}")
    print(f"Absolute Target-10 mIoU Delta      : {delta_target_10:+.4f} ({delta_target_10*100:+.2f}%)")
    print(f"Planar Surfaces Stability Delta    : {delta_planar:+.4f} ({delta_planar*100:+.2f}%)")

    pilot_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_model": model_id,
        "trainable_parameters": trainable_params,
        "frozen_parameters": encoder_params,
        "training_time_sec": round(train_time_sec, 2),
        "peak_train_vram_allocated_mb": round(peak_train_vram_alloc, 1),
        "peak_train_vram_reserved_mb": round(peak_train_vram_res, 1),
        "epoch_logs": epoch_logs,
        "baseline_golden_test_metrics": baseline_metrics,
        "finetuned_golden_test_metrics": post_metrics,
        "delta_target_10_miou": round(delta_target_10, 4),
        "delta_planar_stability": round(delta_planar, 4),
    }

    out_file = REPORTS_DIR / "phase_e_pilot_results.json"
    with open(out_file, "w") as f:
        json.dump(pilot_results, f, indent=2)
    print(f"\nPilot results saved to: {out_file}")


if __name__ == "__main__":
    main()
