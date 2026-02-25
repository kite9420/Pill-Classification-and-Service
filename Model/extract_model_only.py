# extract_model_only.py
# 목적: 다양한 체크포인트 포맷에서 "state_dict만" 추출해서 저장
# 지원:
# - {"model_state_dict": ...} (FP32 training ckpt)
# - {"model_int8_state_dict": ...} (INT8 저장 포맷)
# - state_dict 단독 저장

import os
import torch

IN_PATH  = "checkpoints/epoch_030.pt"
OUT_PATH = "artifacts/model_only.pt"


def main():
    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    obj = torch.load(IN_PATH, map_location="cpu")

    if isinstance(obj, dict):
        if "model_state_dict" in obj:
            state = obj["model_state_dict"]
        elif "model_int8_state_dict" in obj:
            state = obj["model_int8_state_dict"]
        else:
            # dict 자체가 state_dict일 수 있음
            state = obj
    else:
        state = obj

    torch.save(state, OUT_PATH)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()