# compare_4ways.py
# 비교 4종:
# 1) PyTorch FP32 (ckpt)
# 2) PyTorch QAT-INT8 (eager, CPU)
# 3) ONNXRuntime FP32 (onnx)
# 4) ONNXRuntime INT8 (onnx, dynamic)
#
# 출력: acc, avg_loss(CE), throughput(img/s), latency(ms/img)
# ※ 이미지 저장 없음. crop 단위 평가.

import os, json, time
from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision import transforms

import onnxruntime as ort

from model import custom_resnet50_qat
from qat_utils import prepare_model_for_qat, convert_qat_to_int8
from dataset import load_drug_mapping


# ----------------------------
# PATHS (필요시 수정)
# ----------------------------
TEST_LABEL_DIR = "raw_data/Test/label"
TEST_RAW_DIR   = "raw_data/Test/raw"

CKPT_PATH      = "checkpoints/epoch_030.pt"

ONNX_FP32_PATH = "artifacts/model_fp32.onnx"
ONNX_INT8_PATH = "artifacts/model_int8_dynamic.onnx"

MAPPING_JSON   = "artifacts/class_mapping.json"   # drug_to_idx + idx_to_meta
TORCH_INT8_PT  = "artifacts/model_int8.pt"        # quantize_int8_from_qat.py가 만든 파일(선택)

BATCH_SIZE  = 64
NUM_WORKERS = 4


# ----------------------------
# Preprocess (eval과 동일)
# ----------------------------
test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])


# ----------------------------
# JSON/PNG utils
# ----------------------------
def search_json_files(label_dir: str) -> List[str]:
    out = []
    for root, _, files in os.walk(label_dir):
        for f in files:
            if f.endswith(".json"):
                out.append(os.path.join(root, f))
    return out

def read_json(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def build_png_index(raw_dir: str) -> Dict[str, str]:
    idx = {}
    for root, _, files in os.walk(raw_dir):
        for f in files:
            lf = f.lower()
            if lf.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                key = os.path.splitext(f)[0]
                idx.setdefault(key, os.path.join(root, f))
    return idx


# ----------------------------
# Dataset: (png_path, bbox_xyxy_float, true_idx)
# ----------------------------
class TestIndexDataset(Dataset):
    def __init__(self, label_dir: str, raw_dir: str, drug_to_idx: dict):
        self.png_index = build_png_index(raw_dir)
        self.items: List[Tuple[str, Tuple[float,float,float,float], int]] = []

        for jp in tqdm(search_json_files(label_dir), desc="Indexing JSON"):
            data = read_json(jp)

            id_to_fname = {
                int(img["id"]): img.get("file_name")
                for img in data.get("images", [])
                if "id" in img
            }

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


def collate_bboxcrop(batch):
    """
    batch: list[(png_path, (x1,y1,x2,y2), true_idx)]
    같은 png는 1회 open
    return: x(torch.float32 NCHW), y(torch.long)
    """
    grouped = defaultdict(list)
    for pth, bb, yi in batch:
        grouped[pth].append((bb, yi))

    xs, ys = [], []

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
            xs.append(test_tf(crop))
            ys.append(yi)

    x = torch.stack(xs, dim=0)  # float32
    y = torch.tensor(ys, dtype=torch.long)
    return x, y


# ----------------------------
# Torch model loaders
# ----------------------------
def load_torch_fp32(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    num_classes = int(ckpt["num_classes"])
    model = custom_resnet50_qat(num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval().to(device)
    return model, num_classes

def build_torch_int8_from_ckpt(ckpt_path: str, backend: str = "fbgemm"):
    # eager INT8는 CPU용이 기본
    torch.backends.quantized.engine = backend

    ckpt = torch.load(ckpt_path, map_location="cpu")
    num_classes = int(ckpt["num_classes"])

    model_fp32 = custom_resnet50_qat(num_classes=num_classes)
    model_qat = prepare_model_for_qat(model_fp32, backend=backend)
    model_qat.load_state_dict(ckpt["model_state_dict"], strict=False)

    model_int8 = convert_qat_to_int8(model_qat)
    model_int8.eval()
    return model_int8, num_classes

def load_torch_int8_saved(int8_pt_path: str, backend: str = "fbgemm"):
    torch.backends.quantized.engine = backend
    obj = torch.load(int8_pt_path, map_location="cpu")
    num_classes = int(obj["num_classes"])

    # 구조 재생성 → prepare_qat → convert → state load
    model_fp32 = custom_resnet50_qat(num_classes=num_classes)
    model_qat = prepare_model_for_qat(model_fp32, backend=backend)
    model_int8 = convert_qat_to_int8(model_qat)
    model_int8.load_state_dict(obj["model_int8_state_dict"], strict=True)
    model_int8.eval()
    return model_int8, num_classes


# ----------------------------
# ORT session helpers
# ----------------------------
def make_ort_session(onnx_path: str, prefer_cuda: bool = True):
    assert os.path.exists(onnx_path), f"onnx not found: {onnx_path}"
    providers = []
    if prefer_cuda:
        # 환경에 따라 없을 수 있음
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    sess = ort.InferenceSession(onnx_path, providers=providers)
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    return sess, in_name, out_name


# ----------------------------
# Eval runners
# ----------------------------
@torch.no_grad()
def eval_torch(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int):
    crit = nn.CrossEntropyLoss()
    model.eval()

    loss_sum, correct, total = 0.0, 0, 0
    t0 = time.perf_counter()

    for x, y in tqdm(loader, desc="eval(torch)", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = crit(logits, y)

        loss_sum += float(loss.item()) * x.size(0)
        pred = logits.argmax(1)

        correct += int((pred == y).sum().item())
        total += int(x.size(0))

    t1 = time.perf_counter()
    dt = max(1e-9, t1 - t0)

    acc = correct / max(1, total)
    avg_loss = loss_sum / max(1, total)
    ips = total / dt
    ms_per = (dt / max(1, total)) * 1000.0
    return {"acc": acc, "loss": avg_loss, "img_per_s": ips, "ms_per_img": ms_per, "total": total}

def eval_ort(sess: ort.InferenceSession, in_name: str, out_name: str, loader: DataLoader, device_for_loss="cpu"):
    crit = nn.CrossEntropyLoss()

    loss_sum, correct, total = 0.0, 0, 0
    t0 = time.perf_counter()

    for x, y in tqdm(loader, desc="eval(ort)", leave=False):
        # ORT 입력은 numpy float32 NCHW
        x_np = x.numpy().astype(np.float32, copy=False)

        logits_np = sess.run([out_name], {in_name: x_np})[0]  # (N, C)
        logits = torch.from_numpy(logits_np)                  # CPU tensor
        y_cpu = y.to("cpu")

        loss = crit(logits, y_cpu)
        loss_sum += float(loss.item()) * x.size(0)

        pred = logits.argmax(1)
        correct += int((pred == y_cpu).sum().item())
        total += int(x.size(0))

    t1 = time.perf_counter()
    dt = max(1e-9, t1 - t0)

    acc = correct / max(1, total)
    avg_loss = loss_sum / max(1, total)
    ips = total / dt
    ms_per = (dt / max(1, total)) * 1000.0
    return {"acc": acc, "loss": avg_loss, "img_per_s": ips, "ms_per_img": ms_per, "total": total}


def print_result(title: str, r: dict):
    print(f"\n[{title}]")
    print(f"  total      : {r['total']}")
    print(f"  acc        : {r['acc']:.6f}")
    print(f"  loss       : {r['loss']:.6f}")
    print(f"  img/s      : {r['img_per_s']:.2f}")
    print(f"  ms/img     : {r['ms_per_img']:.3f}")


def main():
    assert os.path.isdir(TEST_LABEL_DIR), TEST_LABEL_DIR
    assert os.path.isdir(TEST_RAW_DIR), TEST_RAW_DIR
    assert os.path.exists(MAPPING_JSON), MAPPING_JSON
    assert os.path.exists(CKPT_PATH), CKPT_PATH
    assert os.path.exists(ONNX_FP32_PATH), ONNX_FP32_PATH
    assert os.path.exists(ONNX_INT8_PATH), ONNX_INT8_PATH

    drug_to_idx, idx_to_meta = load_drug_mapping(MAPPING_JSON)
    num_classes = len(idx_to_meta)
    print("num_classes(drug_N):", num_classes)

    ds = TestIndexDataset(TEST_LABEL_DIR, TEST_RAW_DIR, drug_to_idx)
    print("dataset items:", len(ds))
    if len(ds) == 0:
        raise RuntimeError("TestIndexDataset items=0. PNG↔JSON basename 매칭/필터링 확인 필요.")

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,          # ORT도 같이 쓰므로 CPU 이동이 잦음
        collate_fn=collate_bboxcrop,
    )

    # ------------------------
    # 1) Torch FP32 (가능하면 CUDA)
    # ------------------------
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m_fp32, nc1 = load_torch_fp32(CKPT_PATH, device=torch_device)
    if nc1 != num_classes:
        print("[WARN] ckpt num_classes != mapping num_classes:", nc1, num_classes)
    r_torch_fp32 = eval_torch(m_fp32, loader, torch_device, nc1)
    print_result("Torch FP32", r_torch_fp32)

    # ------------------------
    # 2) Torch QAT INT8 (CPU)
    # ------------------------
    backend = "fbgemm"  # x86 윈도우: fbgemm
    if os.path.exists(TORCH_INT8_PT):
        m_int8, nc2 = load_torch_int8_saved(TORCH_INT8_PT, backend=backend)
    else:
        m_int8, nc2 = build_torch_int8_from_ckpt(CKPT_PATH, backend=backend)
    r_torch_int8 = eval_torch(m_int8, loader, torch.device("cpu"), nc2)
    print_result("Torch QAT INT8 (eager, CPU)", r_torch_int8)

    # ------------------------
    # 3) ORT FP32
    # ------------------------
    sess_fp32, in_fp32, out_fp32 = make_ort_session(ONNX_FP32_PATH, prefer_cuda=True)
    r_ort_fp32 = eval_ort(sess_fp32, in_fp32, out_fp32, loader)
    print_result("ONNXRuntime FP32", r_ort_fp32)

    # ------------------------
    # 4) ORT INT8 (dynamic)
    # ------------------------
    # INT8는 보통 CPU EP가 안정적 (CUDA EP에서도 될 수 있으나 환경 의존)
    sess_int8, in_int8, out_int8 = make_ort_session(ONNX_INT8_PATH, prefer_cuda=False)
    r_ort_int8 = eval_ort(sess_int8, in_int8, out_int8, loader)
    print_result("ONNXRuntime INT8 (dynamic)", r_ort_int8)


if __name__ == "__main__":
    main()