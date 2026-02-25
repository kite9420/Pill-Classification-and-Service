# training.py  (FP32 학습 + AMP(autocast+GradScaler) + 후속 QAT/INT8 변환 호환 CKPT 저장)

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from torchvision import transforms

from dataset import build_train_val_crop_from_training_split
from model import custom_resnet50_qat


train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def main():
    # ===== paths =====
    TRAIN_LABEL_DIR = "raw_data/Training/label"
    TRAIN_RAW_DIR   = "raw_data/Training/raw"

    ART_DIR  = "artifacts"
    CKPT_DIR = "checkpoints"

    os.makedirs(ART_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    # ===== device =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # ===== AMP =====
    use_amp = (device.type == "cuda")
    autocast_ctx = torch.cuda.amp.autocast if use_amp else torch.cpu.amp.autocast
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ===== dataset (drug_N label) =====
    train_ds, val_ds, drug_to_idx = build_train_val_crop_from_training_split(
        train_label_dir=TRAIN_LABEL_DIR,
        train_raw_dir=TRAIN_RAW_DIR,
        transform_train=train_tf,
        transform_val=val_tf,
        val_ratio=0.2,
        seed=42,
        mapping_out_json=os.path.join(ART_DIR, "class_mapping.json"),
    )
    num_classes = len(drug_to_idx)
    print("num_classes:", num_classes)

    # ===== dataloader (class imbalance: WeightedRandomSampler) =====
    train_labels = [s[1] for s in train_ds.samples]

    class_count = torch.bincount(torch.tensor(train_labels), minlength=num_classes).float()
    class_count = torch.clamp(class_count, min=1.0)

    sample_weights = (1.0 / class_count)[torch.tensor(train_labels)]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=16,
        sampler=sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    # ===== FP32 model =====
    model = custom_resnet50_qat(num_classes=num_classes).to(device)

    # ===== train setup =====
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    epochs = 30
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = -1.0
    best_path = None

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]")
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast_ctx(enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += float(loss.item()) * x.size(0)
            preds = logits.argmax(dim=1)
            train_correct += int((preds == y).sum().item())
            train_total += int(x.size(0))

            pbar.set_postfix(loss=float(loss.item()), acc=train_correct / max(1, train_total))

        train_loss /= max(1, train_total)
        train_acc = train_correct / max(1, train_total)

        # ---- val ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]")
            for x, y in pbar:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with autocast_ctx(enabled=use_amp):
                    logits = model(x)
                    loss = criterion(logits, y)

                val_loss += float(loss.item()) * x.size(0)
                preds = logits.argmax(dim=1)
                val_correct += int((preds == y).sum().item())
                val_total += int(x.size(0))

                pbar.set_postfix(loss=float(loss.item()), acc=val_correct / max(1, val_total))

        val_loss /= max(1, val_total)
        val_acc = val_correct / max(1, val_total)

        print(
            f"[epoch {epoch}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        # ---- save FP32 ckpt ----
        ckpt_path = os.path.join(CKPT_DIR, f"epoch_{epoch:03d}.pt")
        torch.save(
            {
                "epoch": epoch,
                "num_classes": num_classes,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if use_amp else None,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            },
            ckpt_path,
        )
        print("saved:", ckpt_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = ckpt_path

        scheduler.step()

    print("best_val_acc:", best_val_acc)
    if best_path:
        best_out = os.path.join(CKPT_DIR, "best.pt")
        torch.save(torch.load(best_path, map_location="cpu"), best_out)
        print("saved:", best_out)


if __name__ == "__main__":
    main()