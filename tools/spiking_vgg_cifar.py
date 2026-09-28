"""CIFAR용 Spiking VGG16 — CNN/트랜스포머 스파이크 궤적 대조용.

목적: "학습 중 스파이크가 증가한다"는 관찰이 트랜스포머 고유인지 확인하려면,
CNN을 **같은 조건**으로 재야 한다. TensorFlow-SNNs의 VGG16 실험은 -72%를
기록했지만 그건 다른 구현·다른 계측이다.

그래서 아키텍처를 제외한 모든 것을 QKFormer/Spikformer 실험과 일치시킨다:

  뉴런     MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
  블록     Conv2d(3x3, bias=False) -> BatchNorm2d -> LIF
  시점     T=4, 입력을 T번 반복 (QKFormer와 동일)
  계측     tools/wta_rev.py의 WTARevRegularizer(lam=0) hook
  레시피   cifar10.yml 그대로 (400ep, AdamW, cosine, batch 64, 동일 증강)

spikingjelly의 `spiking_vgg16_bn`을 쓰지 않은 이유: ImageNet 형상이고
(AdaptiveAvgPool 7x7 + 4096 FC) single-step 뉴런이라 [T,B,...] 출력을 전제한
계측 코드와 맞지 않는다. 구조는 표준 CIFAR VGG16이다.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from timm.models.registry import register_model

# 표준 VGG16 (CIFAR): 13 conv + 5 maxpool -> 1x1x512
CFG16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
         512, 512, 512, 'M', 512, 512, 512, 'M']


class ConvBnLif(nn.Module):
    """Conv-BN-LIF. conv/bn은 [T*B,...]에서, LIF는 [T,B,...]에서 동작한다."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, x):                       # [T, B, C, H, W]
        T, B = x.shape[0], x.shape[1]
        y = self.bn(self.conv(x.flatten(0, 1)))
        return self.lif(y.reshape(T, B, *y.shape[1:]))


class SpikingVGG16(nn.Module):
    def __init__(self, num_classes: int = 10, T: int = 4, in_channels: int = 3,
                 width: float = 1.0, pretrained_cfg=None, **kwargs):
        super().__init__()
        self.T = T
        layers, cin = [], in_channels
        for v in CFG16:
            if v == 'M':
                layers.append(nn.MaxPool2d(2))
            else:
                cout = max(1, int(v * width))
                layers.append(ConvBnLif(cin, cout))
                cin = cout
        self.features = nn.ModuleList(layers)
        self.head = nn.Linear(cin, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):                       # [B, C, H, W]
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        for layer in self.features:
            if isinstance(layer, nn.MaxPool2d):
                T, B = x.shape[0], x.shape[1]
                x = layer(x.flatten(0, 1))
                x = x.reshape(T, B, *x.shape[1:])
            else:
                x = layer(x)
        x = x.flatten(3).mean(-1)               # [T, B, C]  공간 평균 (QKFormer와 동일)
        return self.head(x.mean(0))             # 시점 평균 후 분류


@register_model
def spiking_vgg16(pretrained=False, **kwargs):
    kwargs.pop('drop_rate', None)
    kwargs.pop('drop_path_rate', None)
    kwargs.pop('drop_block_rate', None)
    return SpikingVGG16(**kwargs)
