#model.py
import torch
import torch.nn as nn
from typing import List, Optional, Type


import copy
import torch
import torch.nn as nn

import torch.ao.quantization as tq

# ResNet-50 베이스로 Depth를 2,3,4,2로 축소하고 양자화 기능을 추가한 커스텀 모델

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, mid_channels: int, stride: int = 1):
        super().__init__()
        out_channels = mid_channels * self.expansion

        # NOTE: QAT fusion을 위해 Conv/BN/ReLU를 "분리 모듈"로 둔다.
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu_out = nn.ReLU(inplace=True)
        self.skip_add = nn.quantized.FloatFunctional()

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def fuse_model(self):
        # Conv+BN+ReLU fuse
        tq.fuse_modules(self, ["conv1", "bn1", "relu1"], inplace=True)
        tq.fuse_modules(self, ["conv2", "bn2", "relu2"], inplace=True)
        # Conv+BN fuse (마지막은 ReLU가 skip-add 뒤에 있으므로 분리)
        tq.fuse_modules(self, ["conv3", "bn3"], inplace=True)

        if self.downsample is not None:
            # downsample 내부 Conv+BN fuse
            tq.fuse_modules(self.downsample, ["0", "1"], inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.skip_add.add(out, identity)
        out = self.relu_out(out)
        return out


class CustomResNetQAT(nn.Module):
    """
    ResNet-50: layers=[2,3,4,2]로 축소하여 약제 분류에 적합한 크기로 조정.
    7x7 Conv 대신 3x3 Conv 3번으로 Stem 재설계.
    QAT를 위해 QuantStub/DeQuantStub 포함.
    입력 해상도는 AdaptiveAvgPool2d로 변환, 입력 채널 수 3, 출력 클래스 수는 num_classes로 유연하게 설정 가능.
    """
    def __init__(self, layers, num_classes: int = 1000):
        super().__init__()
        self.in_channels = 64

        # QAT entry/exit
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        # Stem (7x7 Conv -> 3x3 Conv 3회로 변경, in channels를 쓰지 않고 계산수 절약을 위해 채널 수를 직접 지정)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stages
        self.layer1 = self._make_layer(Bottleneck, mid_channels=64,  blocks=layers[0], stride=1)
        self.layer2 = self._make_layer(Bottleneck, mid_channels=128, blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(Bottleneck, mid_channels=256, blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(Bottleneck, mid_channels=512, blocks=layers[3], stride=2)

        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        self._init_weights()

        # qconfig는 prepare_qat 전에 지정
        self.qconfig = None

    def _make_layer(self, block, mid_channels: int, blocks: int, stride: int):
        layers = []
        layers.append(block(self.in_channels, mid_channels, stride=stride))
        #각 레이어에 들어가는 채널 수를 in_channels와 block.expansion을 이용해 계산하여 업데이트
        self.in_channels = mid_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, mid_channels, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def fuse_model(self):
        # Stem fuse
        tq.fuse_modules(self, ["conv1", "bn1", "relu"], inplace=True)
        tq.fuse_modules(self, ["conv2", "bn2", "relu2"], inplace=True)
        tq.fuse_modules(self, ["conv3", "bn3", "relu3"], inplace=True)

        # Bottleneck 내부 fuse
        for m in self.modules():
            if isinstance(m, Bottleneck):
                m.fuse_model()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize at entry
        x = self.quant(x)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        # Dequantize at exit (loss 계산/onnx 등)
        x = self.dequant(x)
        return x


def custom_resnet50_qat(num_classes: int):
    return CustomResNetQAT([2, 3, 4, 2], num_classes=num_classes)


