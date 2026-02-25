# quantize_onnx.py  
import os
from onnxruntime.quantization import quantize_dynamic, QuantType

IN_ONNX  = "artifacts/model_fp32.onnx"
OUT_ONNX = "artifacts/model_int8_dynamic.onnx"

def main():
    os.makedirs(os.path.dirname(OUT_ONNX) or ".", exist_ok=True)

    quantize_dynamic(
        model_input=IN_ONNX,
        model_output=OUT_ONNX,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],  # ✅ Conv 제외
        per_channel=False,
    )

    print("saved:", OUT_ONNX)

if __name__ == "__main__":
    main()