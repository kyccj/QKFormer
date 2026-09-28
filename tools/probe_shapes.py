"""LIF 층마다 어떤 모양의 텐서가 오는지 찍는다 (26-09-03).

포팅 전에 두 가지를 확정해야 한다.

1. **dim0 가 정말 T 인가.** `tools/wta_rev.py` 는 `[T,B,...]` 를 가정하고
   `cumsum(dim=0)` 으로 spike_count 를 만든다. 그런데 model.py 를 보면
   `x = x.flatten(0,1)` 뒤에 LIF 를 태우는 자리가 있어(예: cifar10/model.py:117-118)
   dim0 가 T*B 일 수 있다. 그러면 시간이 아니라 배치까지 누적된다.
2. **"채널 내" max 의 축.** 트랜스포머라 `[T,B,C,N]`(토큰) 과 `[T,B,C,H,W]`(공간),
   `[T,B,heads,C/heads,N]` 이 섞인다.

CPU 로 돌리려고 backend 를 'torch' 로 바꾼다 (cupy 는 GPU 전용이고 지금 GPU 가 다 차 있다).
"""
import sys, os
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cifar10'))

import torch
import torch.nn as nn
from timm.models import create_model
from spikingjelly.clock_driven.neuron import MultiStepLIFNode, MultiStepParametricLIFNode

import model as _m   # noqa: F401  (register_model 등록용)

T = 4
net = create_model(
    'QKFormer', pretrained=False,
    drop_rate=0, drop_path_rate=0.1, drop_block_rate=None,
    img_size_h=32, img_size_w=32,
    patch_size=4, embed_dims=384, num_heads=8, mlp_ratios=4,
    in_channels=3, num_classes=10, qkv_bias=False,
    norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=4, sr_ratios=1,
    T=T,
)
net.eval()

TYPES = (MultiStepLIFNode, MultiStepParametricLIFNode)
for m in net.modules():
    if isinstance(m, TYPES):
        m.backend = 'torch'          # cupy 는 GPU 전용

rows = []


def hook(name, module, inp, out):
    if isinstance(out, tuple):
        out = out[0]
    v = getattr(module, 'v_seq', None)
    rows.append((
        name,
        tuple(inp[0].shape) if inp else None,
        tuple(out.shape),
        tuple(v.shape) if torch.is_tensor(v) else None,
        float(module.v_threshold),
        out.shape[0] == T,                       # dim0 가 T 인가
    ))


for name, m in net.named_modules():
    if isinstance(m, TYPES):
        m.register_forward_hook(partial(hook, name))

B = 2
with torch.no_grad():
    net(torch.randn(B, 3, 32, 32))

print(f'T={T}, B={B},  LIF 층 {len(rows)}개\n')
print(f'{"층":42s} {"출력 모양":26s} {"v_seq":26s} {"vth":>5} {"dim0=T":>7}')
bad = []
for name, ishape, oshape, vshape, vth, is_T in rows:
    print(f'{name:42s} {str(oshape):26s} {str(vshape):26s} {vth:5.2f} {"O" if is_T else "X":>7}')
    if not is_T:
        bad.append(name)

print()
if bad:
    print(f'!! dim0 가 T 가 아닌 층 {len(bad)}개 — cumsum(dim=0) 이 배치까지 누적한다:')
    for b in bad:
        print(f'   {b}')
else:
    print('모든 층에서 dim0 == T. cumsum(dim=0) 안전.')

vths = sorted({r[4] for r in rows})
print(f'\nv_threshold 값들: {vths}')
shapes = sorted({len(r[2]) for r in rows})
print(f'출력 차원 수: {shapes}')
for nd in shapes:
    ex = [r[2] for r in rows if len(r[2]) == nd][:3]
    print(f'  {nd}차원 예시: {ex}')
