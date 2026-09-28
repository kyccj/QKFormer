"""flatten(0,1) 로 dim0 가 T·B 가 된 2개 층에서 규제가 얼마나 틀어지는지 잰다 (26-09-03).

`tools/wta_rev.py` 의 `wta_rev_loss` 는 입력이 `[T,B,...]` 라고 보고
  - `cumsum(dim=0)` 으로 spike_count 를 만들고
  - `reshape(T,B,-1)` 뒤 마지막 축에 softmax 를 걸고
  - `reshape(T,-1)` 로 시점당 노름을 낸다.

그런데 `stage3.*.ssa.proj_lif` 두 층은 `model.py:117` 의 `x = x.flatten(0,1)` 때문에
`[T·B, C, N]` 으로 들어온다. 그러면
  - cumsum 이 시간이 아니라 (시간,배치) 를 훑어 **남의 샘플 스파이크가 얹히고**
  - 지역변수 T,B 가 실제로는 (T·B, C) 라 softmax 묶음이 달라지며
  - 노름이 시점당이 아니라 (시점,샘플)당으로 나와 개수가 B배 늘고 크기가 준다.

여기서는 같은 forward 의 같은 텐서에 대해 **현재 계산**과 **[T,B,...] 로 되접은 계산**을
나란히 돌려 sc_rate 와 손실이 얼마나 다른지 숫자로 낸다.
"""
import sys, os
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cifar10'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model
from spikingjelly.clock_driven.neuron import MultiStepLIFNode, MultiStepParametricLIFNode

import model as _m           # noqa: F401
from wta_rev import wta_rev_loss

T, B, ALPHA = 4, 8, 7.0

net = create_model(
    'QKFormer', pretrained=False,
    drop_rate=0, drop_path_rate=0.1, drop_block_rate=None,
    img_size_h=32, img_size_w=32,
    patch_size=4, embed_dims=384, num_heads=8, mlp_ratios=4,
    in_channels=3, num_classes=10, qkv_bias=False,
    norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=4, sr_ratios=1,
    T=T,
)
# 학습된 가중치를 반드시 올린다. 무작위 초기화로 재면 첫 층 뒤로 스파이크가 전혀
# 안 흘러서 손실이 전 층 0 이 되고, 측정이 통째로 무의미해진다 (실제로 그랬다).
CKPT = os.path.expanduser('~/runs/qkformer_wta_rev/lam_0/checkpoint-407.pth.tar')
_c = torch.load(CKPT, map_location='cpu')
missing, unexpected = net.load_state_dict(_c.get('state_dict', _c), strict=False)
print(f'체크포인트: {CKPT}  (epoch {_c.get("epoch")}, metric {_c.get("metric")})')
if missing or unexpected:
    print(f'  missing={len(missing)} unexpected={len(unexpected)}')

net.eval()
TYPES = (MultiStepLIFNode, MultiStepParametricLIFNode)
for m in net.modules():
    if isinstance(m, TYPES):
        m.backend = 'torch'

captured = []


def hook(name, module, inp, out):
    if isinstance(out, tuple):
        out = out[0]
    captured.append((name, out.detach()))


for name, m in net.named_modules():
    if isinstance(m, TYPES):
        m.register_forward_hook(partial(hook, name))

with torch.no_grad():
    net(torch.randn(B, 3, 32, 32) * 0.25)


def sc_rate_of(spike, alpha=ALPHA):
    """wta_rev_loss 와 같은 방식으로 sc_rate 만 뽑는다."""
    t, b = spike.shape[0], spike.shape[1]
    counts = torch.cumsum(spike, dim=0)
    sc_norm = F.softmax((counts / alpha).reshape(t, b, -1), dim=-1).reshape(spike.shape)
    return 1.0 - sc_norm


print(f'T={T}, B={B}, alpha={ALPHA}\n')
print(f'{"층":38s} {"모양":22s} {"손실 현재":>12} {"손실 수정":>12} {"비율":>7} '
      f'{"sc_rate 최대차":>13}')

tot_now = tot_fix = 0.0
rows = []
for name, out in captured:
    now = float(wta_rev_loss(out, ALPHA).item())
    if out.shape[0] == T:                      # 정상 층 — 되접을 것이 없다
        fix, dmax, shape = now, 0.0, tuple(out.shape)
    else:                                      # [T·B, ...] → [T, B, ...]
        fixed = out.reshape(T, -1, *out.shape[1:])
        fix = float(wta_rev_loss(fixed, ALPHA).item())
        a = sc_rate_of(out).reshape(T, -1, *out.shape[1:])
        b_ = sc_rate_of(fixed)
        dmax = float((a - b_).abs().max().item())
        shape = tuple(out.shape)
    tot_now += now
    tot_fix += fix
    rows.append((name, shape, now, fix, dmax))
    flag = '' if out.shape[0] == T else '  <<'
    print(f'{name:38s} {str(shape):22s} {now:12.2f} {fix:12.2f} '
          f'{(fix/now if now else 1):7.3f} {dmax:13.4f}{flag}')

print(f'\n전체 규제 손실   현재 {tot_now:.1f}   수정 {tot_fix:.1f}   '
      f'비율 {tot_fix/tot_now:.4f}  (차이 {100*(tot_fix/tot_now-1):+.2f}%)')

bad = [r for r in rows if r[1][0] != T]
if bad:
    s_now = sum(r[2] for r in bad)
    s_fix = sum(r[3] for r in bad)
    ratio = (s_fix / s_now) if s_now else float('nan')
    print(f'문제의 {len(bad)}개 층만  현재 {s_now:.1f}  수정 {s_fix:.1f}  비율 {ratio:.4f}')
    if tot_now and tot_fix:
        print(f'그 층들이 전체 손실에서 차지하는 비중  현재 {100*s_now/tot_now:.2f}%  '
              f'수정 {100*s_fix/tot_fix:.2f}%')
    print(f'sc_rate 최대 절대차 {max(r[4] for r in bad):.4f}  (sc_rate 는 0~1 범위)')
