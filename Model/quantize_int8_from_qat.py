# quantize_int8_from_qat.py
# 목적:
# 1) FP32 ckpt 로드 (GPU 학습된 모델)
# 2) CPU로 옮겨 QAT 준비(fuse + qconfig + prepare_qat)
# 3) QAT 파인튜닝(몇 epoch) 수행 (FP32, AMP 금지)
# 4) QAT ckpt 저장
# 5) eager INT8(convert) 후 artifacts/model_int8.pt 저장
#
# 주의:
# - eager QAT/INT8는 CPU(fbgemm/qnnpack) 기반이 가장 안정적
# - 여기서는 "CPU QAT fine-tune"을 기본으로 함

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from torchvision import transforms

import torch.ao.quantization as tq

from model import custom_resnet50_qat
from dataset import build_train_val_crop_from_training_split
from qat_utils import prepare_model_for_qat, convert_qat_to_int8


# ===== paths =====
FP32_CKPT_PATH = "checkpoints/epoch_030.pt"

TRAIN_LABEL_DIR = "raw_data/Training/label"
TRAIN_RAW_DIR   = "raw_data/Training/raw"

ART_DIR = "artifacts"
QAT_CKPT_OUT = os.path.join(ART_DIR, "qat_finetuned.pt")     # QAT 상태 저장
INT8_OUT     = os.path.join(ART_DIR, "model_int8.pt")        # eager INT8 저장
MAPPING_JSON = os.path.join(ART_DIR, "class_mapping.json")   # drug_N mapping 저장/재사용

# ===== QAT params =====
BACKEND = "fbgemm"        # x86: fbgemm, arm: qnnpack
EPOCHS = 5                # QAT 파인튜닝 epoch (보통 3~10)
WARMUP_EPOCHS = 1         # observer on 구간
LR = 5e-5                 # QAT는 보통 LR 낮게
WEIGHT_DECAY = 1e-4

BATCH_TRAIN = 16
BATCH_VAL   = 32
NUM_WORKERS = 4


train_tf_qat = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])

val_tf_qat = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])


def build_loaders(num_classes: int):
    train_ds, val_ds, drug_to_idx = build_train_val_crop_from_training_split(
        train_label_dir=TRAIN_LABEL_DIR,
        train_raw_dir=TRAIN_RAW_DIR,
        transform_train=train_tf_qat,
        transform_val=val_tf_qat,
        val_ratio=0.2,
        seed=42,
        mapping_out_json=MAPPING_JSON,
    )
    # class imbalance sampler
    train_labels = [s[1] for s in train_ds.samples]
    class_count = torch.bincount(torch.tensor(train_labels), minlength=num_classes).float()
    class_count = torch.clamp(class_count, min=1.0)
    sample_weights = (1.0 / class_count)[torch.tensor(train_labels)]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_TRAIN,
        sampler=sampler,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,   # CPU QAT
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_VAL,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )
    return train_loader, val_loader


def set_qat_stage(model_qat: nn.Module, epoch: int):
    # warmup: observer ON + fakequant ON
    if epoch <= WARMUP_EPOCHS:
        model_qat.apply(tq.enable_observer)
        model_qat.apply(tq.enable_fake_quant)
    else:
        # 후반: observer OFF + fakequant ON (+ BN freeze)
        model_qat.apply(tq.disable_observer)
        model_qat.apply(tq.enable_fake_quant)
        for m in model_qat.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


@torch.no_grad()
def eval_one(model: nn.Module, loader: DataLoader):
    crit = nn.CrossEntropyLoss()
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in tqdm(loader, desc="val", leave=False):
        x = x.float()  # CPU FP32 강제
        logits = model(x)
        loss = crit(logits, y)
        loss_sum += float(loss.item()) * x.size(0)
        pred = logits.argmax(1)
        correct += int((pred == y).sum().item())
        total += int(x.size(0))
    return loss_sum / max(1, total), correct / max(1, total)


def main():
    os.makedirs(ART_DIR, exist_ok=True)
    assert os.path.exists(FP32_CKPT_PATH), FP32_CKPT_PATH

    # eager quant backend
    torch.backends.quantized.engine = BACKEND

    ckpt = torch.load(FP32_CKPT_PATH, map_location="cpu")
    num_classes = int(ckpt["num_classes"])

    # data
    train_loader, val_loader = build_loaders(num_classes)

    # 1) FP32 모델 복원 (CPU)
    model_fp32 = custom_resnet50_qat(num_classes=num_classes)
    model_fp32.load_state_dict(ckpt["model_state_dict"], strict=True)
    model_fp32.eval()

    # 2) QAT 준비
    model_qat = prepare_model_for_qat(model_fp32, backend=BACKEND)  # deepcopy + fuse + prepare_qat
    # (prepare 전에 로드했으므로, 여기서 추가 load는 불필요)

    # 3) QAT 파인튜닝 (CPU, FP32, AMP 금지)
    model_qat.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model_qat.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        set_qat_stage(model_qat, epoch)

        model_qat.train()
        train_loss_sum, train_correct, train_total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"QAT epoch {epoch}/{EPOCHS} [train]")
        for x, y in pbar:
            x = x.float()  # CPU FP32
            optimizer.zero_grad(set_to_none=True)
            logits = model_qat(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item()) * x.size(0)
            pred = logits.argmax(1)
            train_correct += int((pred == y).sum().item())
            train_total += int(x.size(0))
            pbar.set_postfix(loss=float(loss.item()), acc=train_correct / max(1, train_total))

        train_loss = train_loss_sum / max(1, train_total)
        train_acc = train_correct / max(1, train_total)

        val_loss, val_acc = eval_one(model_qat, val_loader)
        print(f"[QAT epoch {epoch}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model_qat.state_dict().items()}

        scheduler.step()

    # best 복원(선택)
    if best_state is not None:
        model_qat.load_state_dict(best_state, strict=True)

    # 4) QAT ckpt 저장
    torch.save(
        {
            "num_classes": num_classes,
            "backend": BACKEND,
            "qat_epochs": EPOCHS,
            "warmup_epochs": WARMUP_EPOCHS,
            "model_state_dict": model_qat.state_dict(),
            "source_fp32_ckpt": FP32_CKPT_PATH,
            "best_val_acc": float(best_val_acc),
        },
        QAT_CKPT_OUT,
    )
    print("saved:", QAT_CKPT_OUT)

    # 5) INT8 convert + 저장
    model_int8 = convert_qat_to_int8(model_qat).eval()
    torch.save(
        {
            "model_int8_state_dict": model_int8.state_dict(),
            "num_classes": num_classes,
            "backend": BACKEND,
            "source_qat_ckpt": QAT_CKPT_OUT,
            "best_val_acc": float(best_val_acc),
        },
        INT8_OUT,
    )
    print("saved:", INT8_OUT)


if __name__ == "__main__":
    main()