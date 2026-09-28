"""제안법 포팅 점검 (26-09-03).

확인 항목
  1. 되접기가 먹혔는가 — dim0 가 T·B 인 2개 층이 [T,B,...] 로 처리되는가
  2. NaN 이 안 나는가 (QKFormer 는 NaN 전력이 있다 — l2_1e-6_NaN버그 등)
  3. gradient 가 가중치까지 흐르는가
  4. sc_rate 가 [0,1] 을 벗어나지 않는가
  5. 모드별 손실 크기 — rho 를 다시 잡아야 하는 폭을 가늠
"""
import sys, os
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cifar10'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from timm.models import create_model
from spikingjelly.clock_driven.neuron import MultiStepLIFNode, MultiStepParametricLIFNode
from spikingjelly.clock_driven import functional

import model as _m                       # noqa: F401
from wta_rev import WTARevRegularizer, to_TB, prop_loss

T, B = 4, 8
CKPT = os.path.expanduser('~/runs/qkformer_wta_rev/lam_0/checkpoint-407.pth.tar')


def build():
    net = create_model(
        'QKFormer', pretrained=False,
        drop_rate=0, drop_path_rate=0.1, drop_block_rate=None,
        img_size_h=32, img_size_w=32,
        patch_size=4, embed_dims=384, num_heads=8, mlp_ratios=4,
        in_channels=3, num_classes=10, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=4, sr_ratios=1, T=T,
    )
    c = torch.load(CKPT, map_location='cpu')
    net.load_state_dict(c.get('state_dict', c), strict=False)
    for m in net.modules():
        if isinstance(m, (MultiStepLIFNode, MultiStepParametricLIFNode)):
            m.backend = 'torch'
    return net


# --- 1. 되접기 단위 검사 ------------------------------------------------------
x = torch.arange(T * 3 * 2 * 5, dtype=torch.float32).reshape(T, 3, 2, 5)
flat = x.flatten(0, 1)                       # [T*3, 2, 5]
assert torch.equal(to_TB(flat, T), x), '되접기가 원본을 복원하지 못한다'
assert torch.equal(to_TB(x, T), x), '이미 [T,B,...] 인데 건드렸다'
print('1. 되접기 — 복원 정확  OK')

# --- 2~5. 모드별 실행 ---------------------------------------------------------
CASES = [
    ('wta_rev (기존 1-softmax)', dict(mode='wta_rev')),
    ('l2 (기존 plain L2)',       dict(mode='l2')),
    ('prop (제안법)',            dict(mode='prop', vmem_gain=1.0, final_step=True)),
    ('abl_nofinal',              dict(mode='prop', vmem_gain=1.0, final_step=False)),
    ('abl_novmem',               dict(mode='prop', vmem_gain=0.0, final_step=True)),
    ('abl_noinv',                dict(mode='prop', vmem_gain=1.0, final_step=True, invert=False)),
]

print(f'\n{"모드":26s} {"raw 손실":>13} {"grad 노름":>12} {"유한":>5}')
for label, kw in CASES:
    net = build()
    net.train()
    reg = WTARevRegularizer(net, lam=1e-7, T=T, always_raw=True, **kw)
    out = net(torch.randn(B, 3, 32, 32) * 0.25)
    task = out.mean()
    loss = task + reg.loss()
    net.zero_grad()
    loss.backward()
    gnorm = torch.sqrt(sum((p.grad ** 2).sum() for p in net.parameters()
                           if p.grad is not None))
    raw = reg.raw_loss()
    ok = bool(torch.isfinite(torch.tensor(raw)) and torch.isfinite(gnorm))
    print(f'{label:26s} {raw:13.1f} {float(gnorm):12.4f} {"O" if ok else "!!":>5}')
    reg.remove()
    functional.reset_net(net)

# --- sc_rate 범위 ------------------------------------------------------------
print('\nsc_rate 범위 검사 (제안법, 채널 내 maxnorm + vmem)')
net = build(); net.eval()
grabbed = {}


def grab(name, module, inp, out):
    if isinstance(out, tuple):
        out = out[0]
    grabbed[name] = (out.detach(), getattr(module, 'v_seq', None),
                     float(module.v_threshold))


hs = [m.register_forward_hook(partial(grab, n)) for n, m in net.named_modules()
      if isinstance(m, (MultiStepLIFNode, MultiStepParametricLIFNode))]
with torch.no_grad():
    net(torch.randn(B, 3, 32, 32) * 0.25)
for h in hs:
    h.remove()

import torch.nn.functional as F
from wta_rev import _norm_within_channel
lo, hi, worst = 1e9, -1e9, None
for name, (sp, v, vth) in grabbed.items():
    sp = to_TB(sp, T)
    counts = torch.cumsum(sp, 0)[-1].unsqueeze(0)
    vt = to_TB(v, T)[-1].unsqueeze(0)
    readiness = (vt / max(vth, 1e-12)).clamp(0, 1)
    sc = torch.where(counts > 0, counts, 1.0 * readiness)
    r = 1.0 - _norm_within_channel(sc)
    lo, hi = min(lo, float(r.min())), max(hi, float(r.max()))
    if float(r.min()) < -1e-6:
        worst = name
print(f'  sc_rate  최소 {lo:.4f}  최대 {hi:.4f}   (기대 [0, 1])')
print(f'  범위 벗어남: {worst or "없음"}')
print(f'  v_threshold 값들: {sorted({v[2] for v in grabbed.values()})}')
