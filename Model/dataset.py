# dataset.py  (NEW POLICY: category_id 완전 미사용, drug_N 라벨 + dl_name 메타 매핑 유지)

import os, json, random
from typing import Dict, List, Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def _search_json_files(label_dir: str) -> List[str]:
    out = []
    for root, _, files in os.walk(label_dir):
        for f in files:
            if f.endswith(".json"):
                out.append(os.path.join(root, f))
    return out


def _build_png_index(raw_dir: str) -> Dict[str, str]:
    """
    NEW POLICY:
    - raw 평탄/중첩 상관 없이 1회 순회
    - key = basename(확장자 제거)
    """
    idx = {}
    for root, _, files in os.walk(raw_dir):
        for f in files:
            lf = f.lower()
            if lf.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                key = os.path.splitext(f)[0]
                idx.setdefault(key, os.path.join(root, f))
    return idx


def build_and_save_drug_mapping(
    json_paths: List[str],
    out_json_path: str = "class_mapping.json",
):
    """
    NEW POLICY:
    - drug_N 기반 라벨 매핑 + 약 이름 메타까지 같이 저장
    저장 포맷:
      drug_to_idx: {drug_N: idx}
      idx_to_meta: [{idx, drug_N, dl_name, dl_name_en}]
    """
    drug_meta: Dict[str, Dict[str, str]] = {}

    for jp in tqdm(json_paths, desc="Building drug mapping"):
        with open(jp, "r", encoding="utf-8") as f:
            data = json.load(f)

        d = None
        dl_name = ""
        dl_name_en = ""

        for img in data.get("images", []):
            d = img.get("drug_N")
            if d:
                dl_name = str(img.get("dl_name", "") or "")
                dl_name_en = str(img.get("dl_name_en", "") or "")
                break

        if not d:
            continue

        d = str(d)
        # 최초로 본 메타를 보관(중복이면 그대로 유지)
        drug_meta.setdefault(d, {"dl_name": dl_name, "dl_name_en": dl_name_en})

    idx_to_drug = sorted(drug_meta.keys())
    drug_to_idx = {d: i for i, d in enumerate(idx_to_drug)}

    idx_to_meta = [
        {
            "idx": drug_to_idx[d],
            "drug_N": d,
            "dl_name": drug_meta[d].get("dl_name", ""),
            "dl_name_en": drug_meta[d].get("dl_name_en", ""),
        }
        for d in idx_to_drug
    ]

    os.makedirs(os.path.dirname(out_json_path) or ".", exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "drug_to_idx": drug_to_idx,
                "idx_to_meta": idx_to_meta,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return drug_to_idx, idx_to_meta


def load_drug_mapping(json_path: str):
    """
    returns:
      drug_to_idx, idx_to_meta
    """
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    drug_to_idx = {str(k): int(v) for k, v in obj.get("drug_to_idx", {}).items()}
    idx_to_meta = obj.get("idx_to_meta", [])
    return drug_to_idx, idx_to_meta


class PillBboxCropDataset(Dataset):
    """
    annotation(bbox) 1개 = sample 1개
    return: (crop_tensor, label_idx) [+ optional extras]
    label: images[].drug_N 기반 (JSON이 약재 1개라는 전제)
    """
    def __init__(
        self,
        label_dir: str,
        raw_dir: str,
        drug_to_idx: Dict[str, int],
        transform=None,
        return_path: bool = False,
        return_bbox: bool = False,
        json_paths_override: Optional[List[str]] = None,
    ):
        self.transform = transform
        self.return_path = return_path
        self.return_bbox = return_bbox
        self.drug_to_idx = drug_to_idx

        self.json_paths = json_paths_override if json_paths_override is not None else _search_json_files(label_dir)
        self.png_index = _build_png_index(raw_dir)

        # sample: (img_path, label_idx, bbox_xywh, file_name, image_id, ann_id)
        self.samples: List[Tuple[str, int, List[float], str, int, object]] = []

        for jp in tqdm(self.json_paths, desc=f"Scanning(crop) {os.path.basename(label_dir)}"):
            with open(jp, "r", encoding="utf-8") as f:
                data = json.load(f)

            # image_id -> file_name
            id_to_fname = {img.get("id"): img.get("file_name") for img in data.get("images", [])}

            # ---- drug_N 라벨 1번만 뽑기 (JSON=약재 1개 전제) ----
            d = None
            for img in data.get("images", []):
                d = img.get("drug_N")
                if d:
                    break

            if not d:
                continue

            d = str(d)
            label_idx = self.drug_to_idx.get(d, None)
            if label_idx is None:
                continue

            for ann in data.get("annotations", []):
                if ann.get("iscrowd", 0) == 1:
                    continue

                bbox = ann.get("bbox", None)
                if (not bbox) or len(bbox) != 4:
                    continue

                image_id = ann.get("image_id", None)
                if image_id is None:
                    continue

                fname = id_to_fname.get(image_id)
                if not fname:
                    continue

                # NEW POLICY: basename key lookup
                png_key = os.path.splitext(os.path.basename(fname))[0]
                img_path = self.png_index.get(png_key)
                if not img_path:
                    continue

                self.samples.append((img_path, int(label_idx), bbox, fname, int(image_id), ann.get("id", None)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, y, bbox, fname, image_id, ann_id = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        x, y0, w, h = bbox
        W, H = img.size
        x1 = max(0, int(round(x)))
        y1 = max(0, int(round(y0)))
        x2 = min(W, int(round(x + w)))
        y2 = min(H, int(round(y0 + h)))

        if x2 <= x1:
            x2 = min(W, x1 + 1)
        if y2 <= y1:
            y2 = min(H, y1 + 1)

        crop = img.crop((x1, y1, x2, y2)).convert("RGB")

        if self.transform is not None:
            x_t = self.transform(crop)
        else:
            import numpy as np
            x_t = torch.from_numpy(np.array(crop)).permute(2, 0, 1).float() / 255.0

        y_t = torch.tensor(y, dtype=torch.long)

        if self.return_path and self.return_bbox:
            return x_t, y_t, img_path, (x1, y1, x2, y2), fname, image_id, ann_id
        if self.return_path:
            return x_t, y_t, img_path
        if self.return_bbox:
            return x_t, y_t, (x1, y1, x2, y2)
        return x_t, y_t


def build_train_val_crop_from_training_split(
    train_label_dir: str,
    train_raw_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42,
    transform_train=None,
    transform_val=None,
    return_path: bool = False,
    return_bbox: bool = False,
    mapping_out_json: str = "class_mapping.json",
):
    """
    Training/label + Training/raw 만 가지고
    json 파일 단위로 train/val split 후,
    annotation(bbox) 단위 crop dataset 생성.

    NEW POLICY:
      - 라벨은 drug_N만 사용
      - 리턴: (train_ds, val_ds, drug_to_idx)
      - mapping json에는 dl_name/dl_name_en 메타 포함(idx_to_meta)
    """
    json_paths = _search_json_files(train_label_dir)

    drug_to_idx, _ = build_and_save_drug_mapping(
        json_paths=json_paths,
        out_json_path=mapping_out_json,
    )

    rng = random.Random(seed)
    jsons = json_paths[:]
    rng.shuffle(jsons)
    n_val = int(len(jsons) * val_ratio)
    val_jsons = jsons[:n_val]
    train_jsons = jsons[n_val:]

    train_ds = PillBboxCropDataset(
        label_dir=train_label_dir,
        raw_dir=train_raw_dir,
        drug_to_idx=drug_to_idx,
        transform=transform_train,
        return_path=return_path,
        return_bbox=return_bbox,
        json_paths_override=train_jsons,
    )
    val_ds = PillBboxCropDataset(
        label_dir=train_label_dir,
        raw_dir=train_raw_dir,
        drug_to_idx=drug_to_idx,
        transform=transform_val,
        return_path=return_path,
        return_bbox=return_bbox,
        json_paths_override=val_jsons,
    )

    return train_ds, val_ds, drug_to_idx