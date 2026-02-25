# export_onnx.py
import os
import torch

from model import custom_resnet50_qat

CKPT_PATH = "checkpoints/epoch_030.pt"
OUT_ONNX  = "artifacts/model_fp32.onnx"

def main():
    os.makedirs(os.path.dirname(OUT_ONNX) or ".", exist_ok=True)

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    num_classes = int(ckpt["num_classes"])

    model = custom_resnet50_qat(num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    # (선택) fuse 해서 그래프 정리
    # QAT 모델이 fuse_model을 갖고 있으니 가능
    model.fuse_model()
    model.eval()

    dummy = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        OUT_ONNX,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
    )

    print("saved:", OUT_ONNX)

if __name__ == "__main__":
    main()