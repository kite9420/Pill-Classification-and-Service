# qat_utils.py
from __future__ import annotations
import copy
import torch
import torch.nn as nn
import torch.ao.quantization as tq

def prepare_model_for_qat(model: nn.Module, backend: str = "fbgemm") -> nn.Module:
    m = copy.deepcopy(model)

    # fuse는 eval에서
    m.eval()
    m.fuse_model()

    # qconfig
    m.qconfig = tq.get_default_qat_qconfig(backend)

    # prepare_qat는 train에서
    m.train()
    tq.prepare_qat(m, inplace=True)
    return m

@torch.no_grad()
def convert_qat_to_int8(model_qat: nn.Module) -> nn.Module:
    m = copy.deepcopy(model_qat)
    m.eval()
    tq.convert(m, inplace=True)
    return m

def set_qat_stage(model_qat: nn.Module, stage: str):
    """
    stage:
      - "warmup": observer on, fakequant on (기본)
      - "freeze_observer": observer off, fakequant on
      - "freeze_bn": BN stats freeze (선택)
      - "freeze_fakequant": fakequant off (거의 안 씀; 보통 convert 직전에만)
    """
    if stage == "warmup":
        model_qat.apply(tq.enable_observer)
        model_qat.apply(tq.enable_fake_quant)

    elif stage == "freeze_observer":
        model_qat.apply(tq.disable_observer)
        model_qat.apply(tq.enable_fake_quant)

    elif stage == "freeze_fakequant":
        model_qat.apply(tq.disable_fake_quant)

    elif stage == "freeze_bn":
        # BN running stats 고정
        for m in model_qat.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()

    else:
        raise ValueError("unknown stage")