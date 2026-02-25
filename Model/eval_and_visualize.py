# eval_and_visualize.py  (NEW POLICY: category_id 완전 미사용, drug_N 라벨 기반 + I/O 최적화)
import os, json
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

from torchvision import transforms

from model import custom_resnet50_qat
from dataset import load_drug_mapping  # ✅ drug_N 매핑 로더


# ===== paths =====
TEST_LABEL_DIR = "raw_data/Test/label"
TEST_RAW_DIR   = "raw_data/Test/raw"

CKPT_PATH    = "checkpoints/epoch_030.pt"
MAPPING_JSON = "artifacts/class_mapping.json"   # ✅ training에서 저장한 drug_to_idx + idx_to_meta
OUT_DIR      = "Test_result"

BATCH_SIZE = 64
NUM_WORKERS = 4


def collate_bboxcrop(batch):
    """
    Windows spawn 호환: top-level 함수여야 함 (pickle 가능)
    batch: list[(png_path, (x1,y1,x2,y2), true_idx)]
    """
    grouped = defaultdict(list)
    for pth, bb, yi in batch:
        grouped[pth].append((bb, yi))

    xs, ys, pths, bbs = [], [], [], []

    for pth, arr in grouped.items():
        img = Image.open(pth).convert("RGB")
        W, H = img.size

        for (x1f, y1f, x2f, y2f), yi in arr:
            x1 = max(0, min(W - 1, int(round(x1f))))
            y1 = max(0, min(H - 1, int(round(y1f))))
            x2 = max(0, min(W,     int(round(x2f))))
            y2 = max(0, min(H,     int(round(y2f))))
            if x2 <= x1: x2 = min(W, x1 + 1)
            if y2 <= y1: y2 = min(H, y1 + 1)

            crop = img.crop((x1, y1, x2, y2)).convert("RGB")
            xs.append(test_tf(crop) if test_tf is not None else transforms.ToTensor()(crop))
            ys.append(yi)
            pths.append(pth)
            bbs.append((x1, y1, x2, y2))

    x = torch.stack(xs, dim=0)
    y = torch.tensor(ys, dtype=torch.long)
    bb = torch.tensor(bbs, dtype=torch.int32)
    return x, y, pths, bb


# ===== utils =====
def search_json_files(label_dir: str):
    out = []
    for root, _, files in os.walk(label_dir):
        for f in files:
            if f.endswith(".json"):
                out.append(os.path.join(root, f))
    return out


def read_json(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def build_png_index(raw_dir: str):
    # NEW POLICY: key=basename
    idx = {}
    for root, _, files in os.walk(raw_dir):
        for f in files:
            lf = f.lower()
            if lf.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                key = os.path.splitext(f)[0]
                idx.setdefault(key, os.path.join(root, f))
    return idx


def font(size=16):
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])


class TestIndexDataset(Dataset):
    """
    ✅ 인덱싱 전용: (png_path, bbox_xyxy_float, true_idx)
    - 이미지 파일은 여기서 열지 않음
    - true_idx는 drug_N 기반
    """
    def __init__(self, label_dir: str, raw_dir: str, drug_to_idx: dict):
        self.png_index = build_png_index(raw_dir)
        self.items = []  # (png_path, (x1,y1,x2,y2), true_idx)

        for jp in tqdm(search_json_files(label_dir), desc="Indexing JSON"):
            data = read_json(jp)

            # image_id -> file_name
            id_to_fname = {
                int(img["id"]): img.get("file_name")
                for img in data.get("images", [])
                if "id" in img
            }

            # ✅ drug_N 라벨 1회만 추출 (JSON=약재 1개 전제)
            drug_name = None
            for img in data.get("images", []):
                drug_name = img.get("drug_N")
                if drug_name:
                    break
            if not drug_name:
                continue

            drug_name = str(drug_name)
            if drug_name not in drug_to_idx:
                continue
            true_idx = int(drug_to_idx[drug_name])

            for ann in data.get("annotations", []):
                if ann.get("iscrowd", 0) == 1:
                    continue
                bbox = ann.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue

                image_id = ann.get("image_id")
                if image_id is None:
                    continue
                image_id = int(image_id)

                fname = id_to_fname.get(image_id)
                if not fname:
                    continue

                # NEW POLICY basename lookup
                key = os.path.splitext(os.path.basename(fname))[0]
                png_path = self.png_index.get(key)
                if not png_path:
                    continue

                x, y, w, h = bbox
                self.items.append((png_path, (x, y, x + w, y + h), true_idx))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


@torch.no_grad()
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # ✅ drug mapping 로드
    drug_to_idx, idx_to_meta = load_drug_mapping(MAPPING_JSON)
    num_classes = len(idx_to_meta)
    print("num_classes(drug_N):", num_classes)

    ds = TestIndexDataset(TEST_LABEL_DIR, TEST_RAW_DIR, drug_to_idx)

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_bboxcrop,  # ✅ 이미지 open/crop을 여기서 통제
    )

    model = custom_resnet50_qat(num_classes=num_classes).to(device).eval()
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    crit = nn.CrossEntropyLoss()

    loss_sum, correct, total = 0.0, 0, 0
    per_img = defaultdict(list)  # img_path -> list[(bbox, pred_idx, true_idx)]

    for x, y, pths, bbs in tqdm(loader, desc="Eval+Collect"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = crit(logits, y)

        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(1)

        correct += (pred == y).sum().item()
        total += x.size(0)

        pred_cpu = pred.detach().cpu().tolist()
        y_cpu    = y.detach().cpu().tolist()
        bb_cpu   = bbs.detach().cpu().tolist()

        for pth, bb, pi, yi in zip(pths, bb_cpu, pred_cpu, y_cpu):
            per_img[pth].append((tuple(map(int, bb)), int(pi), int(yi)))

    print("crop_test_loss:", loss_sum / max(1, total))
    print("crop_test_acc :", correct / max(1, total))

    fnt = font(16)

    # ✅ 저장 단계: 이미지당 open 1회
    for pth, items in tqdm(per_img.items(), desc="Saving images"):
        base = Image.open(pth).convert("RGB")
        draw = ImageDraw.Draw(base)
        W, H = base.size

        for (x1, y1, x2, y2), pred_idx, true_idx in items:
            pm = idx_to_meta[pred_idx] if 0 <= pred_idx < len(idx_to_meta) else {}
            tm = idx_to_meta[true_idx] if 0 <= true_idx < len(idx_to_meta) else {}

            pred_drug = pm.get("drug_N", "")
            pred_name = pm.get("dl_name", "")
            true_drug = tm.get("drug_N", "")
            true_name = tm.get("dl_name", "")

            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)

            text = (
                f"PRED idx:{pred_idx} {pred_drug} {pred_name}\n"
                f"TRUE idx:{true_idx} {true_drug} {true_name}"
            )

            tb = draw.multiline_textbbox((0, 0), text, font=fnt)
            tw = tb[2] - tb[0]
            th = tb[3] - tb[1]
            pad = 3

            tx1 = x1
            ty1 = max(0, y1 - th - 2 * pad)
            tx2 = min(W, x1 + tw + 2 * pad)
            ty2 = ty1 + th + 2 * pad

            draw.rectangle([tx1, ty1, tx2, ty2], fill=(0, 0, 0))
            draw.multiline_text((tx1 + pad, ty1 + pad), text, fill=(255, 255, 255), font=fnt)

        out_name = os.path.splitext(os.path.basename(pth))[0] + ".png"
        base.save(os.path.join(OUT_DIR, out_name))

    print("done:", OUT_DIR)


if __name__ == "__main__":
    main()