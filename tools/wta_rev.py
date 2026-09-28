"""EIP wta_rev 스파이크 규제 — TensorFlow-SNNs 구현의 PyTorch 포팅.

원본: TensorFlow-SNNs `lib_snn/neurons.py` (sc_rate 계산) +
      `lib_snn/layers.py::l2_norm_wta_rev` (수정된 gradient)

층·시점마다:

    sc      = cumsum_t(spike) / alpha          # spike_count, stop-gradient
    sc_norm = softmax(sc)                       # 배치 외 전 차원에 대해
    sc_rate = 1 - sc_norm                       # 적게 발화한 뉴런일수록 큰 가중치
    x       = spike * sc_rate
    loss_t  = l2_norm_wta_rev(x, sc_rate)
    total  += lambda * sum_t loss_t

`l2_norm_wta_rev`의 핵심은 forward는 보통의 L2 norm이지만 backward에서
`x/||x||` 대신 `sc_rate/||x||`를 쓴다는 것이다. 표준 L2는 spike=0인 뉴런에
gradient가 0으로 도달하는데(x가 0이므로), 이 변형은 발화하지 않은 뉴런에도
gradient를 보낸다. "발화하지 마라"는 신호를 loser에게 전달하려는 설계다.

다만 `analysis.md`에 기록된 대로 surrogate gradient가 threshold에서 멀면
0에 가까우므로, 이 우회가 실제로 효과가 있는지는 별개 문제다. 이 포팅은
그 검증을 QKFormer(트랜스포머 계열)에서 수행하기 위한 것이다.

원본과의 차이 (의도된 것):
  TF는 시점마다 호출되며 self.spike_count가 그때까지 누적된 값이다.
  PyTorch의 MultiStepLIFNode는 [T,B,...]를 한 번에 내므로, 동일한 누적을
  cumsum(dim=0)으로 재현한다. 현재 시점의 스파이크를 포함하는 inclusive
  누적이다.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from spikingjelly.clock_driven.neuron import (
        MultiStepLIFNode,
        MultiStepParametricLIFNode,
    )
    SPIKING_TYPES = (MultiStepLIFNode, MultiStepParametricLIFNode)
except ImportError:  # pragma: no cover
    SPIKING_TYPES = ()


class _L2NormWTARev(torch.autograd.Function):
    """시점축을 한 번에 처리하는 wta_rev L2 norm.

    입력 [T,B,...]에 대해 시점별 L2 norm [T]를 반환한다. backward는 시점 t의
    x가 아니라 sc_rate를 쓴다 — 이것이 비발화 뉴런에도 gradient가 가는 이유다.

    norm==0인 시점은 gradient 0 (원본의 count_nonzero == 0 분기). 이 판정을
    `.item()` 대신 마스크로 하는 것이 중요하다. `.item()`은 GPU 동기화를
    강제해서 층 35개 x 시점 4개 = 140회 동기화가 발생했다.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, sc_rate: torch.Tensor):
        T = x.shape[0]
        norms = torch.sqrt((x * x).reshape(T, -1).sum(-1))     # [T]
        ctx.save_for_backward(sc_rate, norms)
        return norms

    @staticmethod
    def backward(ctx, grad_out):                                # grad_out: [T]
        sc_rate, norms = ctx.saved_tensors
        shape = (-1,) + (1,) * (sc_rate.dim() - 1)
        zero = norms == 0
        safe = norms.masked_fill(zero, 1.0)
        coef = (grad_out / safe).masked_fill(zero, 0.0).reshape(shape)
        return coef * sc_rate, None


class _L2NormWTARevW1(torch.autograd.Function):
    """w¹ 기울기판 (26-09-25). TF `lib_snn/layers.py::l2_norm_wta_rev_w1` 대응.

    기존 호출은 `_L2NormWTARev.apply(pen * sc_rate, sc_rate)` 라 곱셈이 Function
    **바깥**에 있다. autograd 가 그 곱셈을 한 번 더 미분하므로
    ∂L/∂pen = sc_rate/‖x‖ · sc_rate = sc_rate²/‖x‖ (w²) 가 된다.
    여기서는 곱셈을 forward **안**에서 하고 backward 에서 pen 에 대한 기울기를
    직접 돌려주므로 sc_rate 가 한 번만 곱해진다: ∂L/∂pen = sc_rate/‖x‖ (w¹).

    forward 는 `_L2NormWTARev` 와 같은 연산을 같은 순서로 하므로 값이 비트 단위로
    같다 — loss-ratio 의 R(raw 손실) 과 ρ 눈금을 w² 런과 공유하려면 이래야 한다.
    """

    @staticmethod
    def forward(ctx, pen: torch.Tensor, sc_rate: torch.Tensor):
        x = pen * sc_rate                                        # 안에서 곱한다
        T = x.shape[0]
        norms = torch.sqrt((x * x).reshape(T, -1).sum(-1))     # [T] — 기존과 동일
        ctx.save_for_backward(sc_rate, norms)
        return norms

    @staticmethod
    def backward(ctx, grad_out):                                # grad_out: [T]
        sc_rate, norms = ctx.saved_tensors
        shape = (-1,) + (1,) * (sc_rate.dim() - 1)
        zero = norms == 0
        safe = norms.masked_fill(zero, 1.0)
        coef = (grad_out / safe).masked_fill(zero, 0.0).reshape(shape)
        return coef * sc_rate, None                             # pen 기울기 — sc_rate 한 번


def _softmax_over_non_batch(x: torch.Tensor) -> torch.Tensor:
    """배치 차원을 제외한 전 차원에 대한 softmax."""
    b = x.shape[0]
    return F.softmax(x.reshape(b, -1), dim=-1).reshape(x.shape)


def plain_l2_loss(spike: torch.Tensor) -> torch.Tensor:
    """비교군: 스파이크에 대한 표준 L2. WTA 가중도, gradient 수정도 없다.

    TensorFlow-SNNs의 `analysis.md`는 CNN에서 WTA 변형 8종이 전부 이것과 같은
    accuracy-spike tradeoff로 수렴한다고 기록했다. 트랜스포머에서도 같은지 본다.

    수학적으로 wta_rev와 forward가 거의 같다 — N이 크면 softmax가 1/N에 가까워
    sc_rate = 1 - sc_norm ≈ 1이 되기 때문이다. 차이는 backward에만 있다
    (표준 x/||x|| vs 수정된 sc_rate/||x||). 따라서 같은 lambda로 직접 비교된다.
    """
    T = spike.shape[0]
    sq = (spike * spike).reshape(T, -1).sum(-1)          # [T]
    # sqrt(0)의 미분은 발산한다. d/dx sqrt(x) = 1/(2 sqrt(x))가 x=0에서 inf이고,
    # 여기에 dx/ds = 2s = 0이 곱해져 inf * 0 = NaN이 된다. 발화가 전혀 없는
    # 층/시점이 하나라도 있으면 전체 gradient가 오염되고, AMP의 GradScaler가
    # 매 스텝을 건너뛰어 학습이 완전히 멈춘다 (실제로 409 epoch 동안 그랬다).
    # wta_rev는 backward에서 norm==0을 마스크로 막지만 여기는 표준 autograd라
    # forward에서 막아야 한다.
    zero = sq == 0
    return torch.sqrt(sq.masked_fill(zero, 1.0)).masked_fill(zero, 0.0).sum()


def wta_rev_loss(
    spike: torch.Tensor,
    alpha: float = 7.0,
    rate_alpha: float = 1.0,
) -> torch.Tensor:
    """[T, B, ...] 스파이크 텐서에 대한 wta_rev 규제 손실 (lambda 미적용).

    Args:
        spike: MultiStepLIFNode 출력. [T, B, ...], 값은 {0, 1}.
        alpha: softmax 온도 (원본 `reg_spike_out_alpha`, 실험에서 7).
        rate_alpha: sc_rate 배율 (원본 `reg_spike_rate_alpha`, 기본 1).
    """
    T, B = spike.shape[0], spike.shape[1]
    # spike_count 누적. stop-gradient — 가중치이지 학습 대상이 아니다.
    counts = torch.cumsum(spike.detach(), dim=0)

    # 배치 외 전 차원에 대한 softmax를 전 시점에 한 번에. (t, b)마다 독립이므로
    # [T,B,-1]로 펴서 마지막 축에 softmax를 걸면 시점 루프와 동일하다.
    sc_norm = F.softmax((counts / alpha).reshape(T, B, -1), dim=-1).reshape(spike.shape)
    sc_rate = (1.0 - sc_norm) * rate_alpha
    return _L2NormWTARev.apply(spike * sc_rate, sc_rate).sum()


# ---------------------------------------------------------------------------
# 제안법 (26-09-03 추가) — TF `reg_spike_final_step` + vmem + 채널 내 1-maxnorm
# ---------------------------------------------------------------------------

def to_TB(x: torch.Tensor, T: int) -> torch.Tensor:
    """dim0 가 T·B 로 합쳐진 텐서를 [T, B, ...] 로 되돌린다.

    QKFormer 는 `cifar10/model.py:117` 에서 `x = x.flatten(0, 1)` 을 한 뒤 LIF 를
    태우는 자리가 있어서, 35개 층 중 `stage3.*.ssa.proj_lif` 2개가 `[T·B, C, N]`
    로 들어온다. 그대로 두면 `cumsum(dim=0)` 이 시간이 아니라 (시간,배치) 를 훑어
    **남의 샘플 스파이크가 count 에 얹히고**, 노름도 시점당이 아니라 (시점,샘플)당
    으로 나와 개수가 B배 늘어난다.

    실측 (학습된 lam_0 체크포인트, T=4 B=8):
        그 2개 층의 규제 손실이 2.75배 과대 (2821.9 -> 1024.2)
        sc_rate 최대 절대차 0.155 (sc_rate 는 0~1 범위)
        전체 규제 손실로는 -5.87%
    배치가 커질수록 오염이 커지므로 실제 학습(B=64)에서는 이보다 크다.

    `flatten(0,1)` 은 메모리 순서를 안 바꾸고 축만 합치므로 (인덱스 t·B + b),
    `reshape(T, -1, ...)` 로 손실 없이 정확히 복원된다.
    """
    if x.shape[0] == T:
        return x
    if x.shape[0] % T != 0:
        raise RuntimeError(f'dim0={x.shape[0]} 이 T={T} 의 배수가 아니다 — 모양 가정이 깨졌다')
    return x.reshape(T, -1, *x.shape[1:])


def _norm_within_channel(sc: torch.Tensor) -> torch.Tensor:
    """[T,B,C,...] 에서 dim2 를 채널로 보고 그 뒤 전 축에 max 를 걸어 나눈다.

    TF 의 `reg_spike_maxnorm_group='within_channel'` 대응 — 거기서는 [B,H,W,C] 의
    공간축(H,W) 에 max 였다. QKFormer 는 채널이 dim2 이고 그 뒤가 공간(H,W) 또는
    토큰(N) 이라 "dim3 이후 전부" 가 같은 역할을 한다.

    5차원 `[T,B,heads,C//heads,N]`(stage1·2 의 attn_lif) 은 dim2 가 head 다.
    C//heads=1 이라 사실상 `[T,B,heads,N]` 이고, head 마다 따로 max 를 잡는 것이
    채널별 max 와 같은 취급이 된다.

    분모가 0 인 곳(그 채널이 통째로 침묵)은 TF 의 `divide_no_nan` 과 같게 0 을 준다.
    """
    red = tuple(range(3, sc.dim())) if sc.dim() > 3 else (2,)
    m = sc.amax(dim=red, keepdim=True)
    return torch.where(m > 0, sc / m.clamp_min(1e-12), torch.zeros_like(sc))


def _norm_layer(sc: torch.Tensor) -> torch.Tensor:
    """[T,B,...] 에서 (t,b) 마다 **dim2 이후 전 축에 max 하나**를 잡아 나눈다 (26-09-25).

    TF 의 `reg_spike_maxnorm_group='none'`(= `ours_layer`) 대응 — 층의 모든 뉴런이
    한 샘플 안에서 서로 겨룬다. `_norm_within_channel` 과 달리 "채널"이 무엇인지
    (dim2 가 채널이든 head 든, 뒤가 공간이든 토큰이든) **상관없다**: 샘플당 전체
    max 한 개다. 따라서 1-maxnorm 에서 sc_rate=0 (면제) 은 샘플당 1등 뉴런(동점이면
    동점 수만큼) 뿐이다.

    배치(dim1)는 섞지 않는다 — 샘플마다 독립. 시점(dim0)도 섞지 않는다 —
    final_step 이면 dim0 길이가 1 이고, 아니면 시점마다 따로 max 를 잡는 것이
    within_channel 과 같은 취급이다.
    """
    if sc.dim() < 3:
        raise RuntimeError(f'sc 는 [T,B,...] 3차원 이상이어야 한다. 받은 모양: {tuple(sc.shape)}')
    red = tuple(range(2, sc.dim()))
    m = sc.amax(dim=red, keepdim=True)
    return torch.where(m > 0, sc / m.clamp_min(1e-12), torch.zeros_like(sc))


def prop_loss(
    spike: torch.Tensor,
    v_seq: torch.Tensor = None,
    v_threshold: float = 1.0,
    T: int = 4,
    vmem_gain: float = 0.0,
    final_step: bool = True,
    maxnorm: bool = True,
    invert: bool = True,
    alpha: float = 7.0,
    rate_alpha: float = 1.0,
    maxnorm_group: str = 'within_channel',
    grad: str = 'w2',
) -> torch.Tensor:
    """제안법 규제 손실 (lambda 미적용).

        sc_soft = spike_count          (발화한 뉴런)
                = gain · clip(vmem/vth, 0, 1)   (한 번도 발화 안 한 뉴런)
        sc_rate = 1 − sc_soft / max_채널(sc_soft)
        loss    = ‖ (Σ_t spike_t) · sc_rate ‖   ← final_step 이면 t=T 에 한 번만

    Args:
        spike: LIF 출력. `[T,B,...]` 또는 `[T·B,...]` (후자는 되접는다).
        v_seq: 같은 모양의 막전위. spikingjelly 의 `module.v_seq` — 이 버전은
            torch/cupy 두 백엔드 모두 항상 채우므로 플래그가 필요 없다.
            리셋 **후** 값이라 TF 에서 쓰는 것과 의미가 같다.
        v_threshold: 그 모듈의 문턱. QKFormer 는 `attn_lif` 만 0.5 이고 나머지는
            1.0 이라 **전역 1.0 으로 나누면 그 층만 틀린다.** 모듈별로 받는다.
        final_step: True 면 t=T 에서 한 번만 건다. 손실이 받는 양이
            `Σ_t‖·‖`(T개 항) 에서 `‖Σ_t ·‖`(1개 항) 으로 바뀌므로 실효 lambda 가
            달라진다 — rho 를 다시 잡아야 한다.
        maxnorm: True 면 채널 내 max 로 정규화(1-maxnorm), False 면 softmax(1-softmax).
        invert: `1 −` 반전. False 면 sc_rate = sc_norm (많이 쏘는 쪽을 벌한다).
        maxnorm_group: maxnorm 의 경쟁 범위 (26-09-25). TF `reg_spike_maxnorm_group`.
            'within_channel' (기본, 기존 동작) = 채널(dim2)마다 max → `ours_intra_ch`.
            'none' = 샘플당 층 전체 max 하나 → `ours_layer`. maxnorm=False 면 무시.
        grad: 'w2' (기본, 기존 동작) 면 ∂/∂pen = sc_rate²/‖x‖, 'w1' 이면 sc_rate/‖x‖.
            forward 값은 둘이 같다. `_L2NormWTARevW1` 설명 참조.
    """
    if maxnorm_group not in ('within_channel', 'none'):
        raise ValueError(f"maxnorm_group 은 'within_channel' / 'none'. 받은 값: {maxnorm_group}")
    if grad not in ('w2', 'w1'):
        raise ValueError(f"grad 는 'w2' / 'w1'. 받은 값: {grad}")
    spike = to_TB(spike, T)
    counts = torch.cumsum(spike.detach(), dim=0)          # [T,B,...] 시점별 누적

    if final_step:
        sc = counts[-1]                                    # 최종 카운트 [B,...]
        pen = spike.sum(0)                                 # 미분 가능한 스파이크 합
        v_t = to_TB(v_seq, T)[-1] if (vmem_gain > 0.0 and v_seq is not None) else None
        sc, pen = sc.unsqueeze(0), pen.unsqueeze(0)        # [1,B,...] 로 축 맞춤
        if v_t is not None:
            v_t = v_t.unsqueeze(0)
    else:
        sc, pen = counts, spike
        v_t = to_TB(v_seq, T) if (vmem_gain > 0.0 and v_seq is not None) else None

    # 침묵 뉴런만 막전위로 가른다. 발화한 뉴런은 정수 카운트를 그대로 유지해야
    # 활동량 순위가 안 뒤집힌다 (gain<=1 이면 침묵의 최대가 1 을 못 넘는다).
    if v_t is not None:
        readiness = (v_t.detach() / max(v_threshold, 1e-12)).clamp(0.0, 1.0)
        sc = torch.where(sc > 0, sc, vmem_gain * readiness)

    if maxnorm:
        sc_norm = _norm_layer(sc) if maxnorm_group == 'none' else _norm_within_channel(sc)
    else:
        t_, b_ = sc.shape[0], sc.shape[1]
        sc_norm = F.softmax((sc / alpha).reshape(t_, b_, -1), dim=-1).reshape(sc.shape)

    # sc_rate 는 detach 된 counts·v_t 에서만 나오므로 기울기가 끊겨 있다 (w1/w2 공통).
    sc_rate = ((1.0 - sc_norm) if invert else sc_norm) * rate_alpha
    if grad == 'w1':
        return _L2NormWTARevW1.apply(pen, sc_rate).sum()
    return _L2NormWTARev.apply(pen * sc_rate, sc_rate).sum()


class WTARevRegularizer:
    """모델 전 스파이킹 층에 wta_rev 규제를 건다. 모델 코드는 수정하지 않는다.

        reg = WTARevRegularizer(model, lam=1e-7)
        ...
        output = model(input)
        loss = loss_fn(output, target) + reg.loss()
        loss.backward()
        reg.reset()            # functional.reset_net과 같은 자리에서
    """

    def __init__(
        self,
        model: nn.Module,
        lam: float,
        alpha: float = 7.0,
        rate_alpha: float = 1.0,
        types=None,
        mode: str = 'wta_rev',
        always_raw: bool = False,
        T: int = 4,
        vmem_gain: float = 0.0,
        final_step: bool = False,
        invert: bool = True,
        maxnorm_group: str = 'within_channel',
        grad: str = 'w2',
    ):
        if mode not in ('wta_rev', 'l2', 'prop'):
            raise ValueError(f"mode는 'wta_rev' / 'l2' / 'prop'. 받은 값: {mode}")
        self.lam = lam
        self.alpha = alpha
        self.rate_alpha = rate_alpha
        self.mode = mode
        # 'prop' 계열 손잡이. mode='prop' 일 때만 쓰인다.
        #   vmem_gain=0 -> 침묵 뉴런 무차등 (abl_novmem)
        #   final_step=False -> 매 시점 규제 (abl_nofinal)
        #   invert=False -> 1- 반전 제거 (abl_noinv)
        self.T = T
        self.vmem_gain = vmem_gain
        self.final_step = final_step
        self.invert = invert
        #   maxnorm_group='none' -> 층 전체 경쟁 (ours_layer). 기본은 채널 내 (ours_intra_ch)
        #   grad='w1' -> sc_rate 를 한 번만 곱한 기울기. 기본 w2 는 기존(TF 버그와 동일)
        if maxnorm_group not in ('within_channel', 'none'):
            raise ValueError(f"maxnorm_group 은 'within_channel' / 'none'. 받은 값: {maxnorm_group}")
        if grad not in ('w2', 'w1'):
            raise ValueError(f"grad 는 'w2' / 'w1'. 받은 값: {grad}")
        self.maxnorm_group = maxnorm_group
        self.grad = grad
        # loss-ratio 제어는 lambda=0에서 출발하는데, raw 값이 없으면 비율을 계산할 수
        # 없어 lambda가 0에 갇힌다. 그래서 lambda와 무관하게 손실을 계산해 둔다
        # (loss()는 lambda를 곱하므로 lambda=0이면 학습에 영향 없음).
        self.always_raw = always_raw
        self.types = types or SPIKING_TYPES
        if not self.types:
            raise RuntimeError("spikingjelly 뉴런 타입을 찾을 수 없다.")

        self._handles = []
        self._losses: List[torch.Tensor] = []
        self.layers: List[str] = []
        # Pareto 비교용 스파이크 집계 (규제와 무관, detach된 값)
        self._spikes = None          # GPU 텐서로 누적
        self._samples = 0
        self._counted_this_forward = False

        for name, module in model.named_modules():
            if isinstance(module, self.types):
                self.layers.append(name)
                self._handles.append(module.register_forward_hook(self._hook))
        if not self._handles:
            raise RuntimeError("스파이킹 뉴런 모듈을 찾지 못했다.")

    def _hook(self, _module, _inp, out):
        if isinstance(out, tuple):
            out = out[0]
        module = _module
        # 스파이크 집계는 항상 한다. 규제가 꺼져 있어도 baseline 궤적이 필요하다.
        # GPU에 누적하고 읽을 때만 동기화한다 (.item()을 층마다 부르면 35회 동기화).
        s = out.detach().sum()
        self._spikes = s if self._spikes is None else self._spikes + s
        if not self._counted_this_forward:
            self._samples += out.shape[1] if out.dim() > 1 else 1
            self._counted_this_forward = True
        if not torch.is_grad_enabled():
            return
        if self.lam == 0.0 and not self.always_raw:
            return
        # dim0 가 T·B 로 합쳐진 층(stage3.*.ssa.proj_lif 2개)을 [T,B,...] 로 되돌린다.
        # 안 하면 cumsum 이 배치까지 누적하고 노름 개수가 B배로 늘어난다.
        # 실측: 그 2층 손실 2.75배 과대, sc_rate 최대차 0.155, 전체로는 -5.87%.
        # 세 mode 전부에 적용해야 층 간 배율이 같아진다.
        out = to_TB(out, self.T)

        if self.mode == 'l2':
            self._losses.append(plain_l2_loss(out))
        elif self.mode == 'prop':
            # v_seq 는 spikingjelly 가 항상 채운다 (torch/cupy 둘 다). 별도 플래그 불필요.
            # v_threshold 는 모듈마다 다르다 — QKFormer 는 attn_lif 만 0.5 다.
            self._losses.append(prop_loss(
                out,
                v_seq=getattr(module, 'v_seq', None),
                v_threshold=float(getattr(module, 'v_threshold', 1.0)),
                T=self.T,
                vmem_gain=self.vmem_gain,
                final_step=self.final_step,
                maxnorm=True,
                invert=self.invert,
                alpha=self.alpha,
                rate_alpha=self.rate_alpha,
                maxnorm_group=self.maxnorm_group,
                grad=self.grad,
            ))
        else:
            self._losses.append(wta_rev_loss(out, self.alpha, self.rate_alpha))

    def spikes_per_sample(self) -> float:
        """누적 구간의 샘플당 총 스파이크 수. epoch 단위 집계에 쓴다."""
        if self._spikes is None or not self._samples:
            return 0.0
        return self._spikes.item() / self._samples      # 여기서만 동기화

    def reset_spike_stats(self) -> None:
        self._spikes = None
        self._samples = 0

    def loss(self) -> torch.Tensor:
        """이번 forward의 규제 손실 (lambda 적용). 없으면 0."""
        if not self._losses:
            return torch.zeros((), requires_grad=False)
        return self.lam * torch.stack(self._losses).sum()

    def raw_loss(self) -> float:
        """lambda 적용 전 값. task loss와의 비율을 보는 용도."""
        if not self._losses:
            return 0.0
        return torch.stack(self._losses).sum().item()

    def reset(self) -> None:
        self._losses.clear()
        self._counted_this_forward = False

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
