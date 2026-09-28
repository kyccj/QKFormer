"""TF 저장소의 asym 대체 기울기를 SpikingJelly 로 옮긴 것 (26-09-07).

**왜 필요한가** — TensorFlow-SNNs 쪽 실험은 전부 `fire_surro_grad_func='asym'` 으로 돌고 있다
(`config_snn_training.py:172`, 조건 없이 무조건 켜짐). QKFormer 는 SpikingJelly 기본값인
`surrogate.Sigmoid()` 를 쓴다. 두 코드베이스 결과를 한 표에 올리려면 대체 기울기를 맞춰야 한다.

**TF 원본** (`lib_snn/neurons.py:2103-2114`):

    width_h = conf.surro_grad_alpha            # 기본 0.5
    cond    = (vth - width_h <= vmem <= vth + width_h)
    h       = (1 - bias) * vmem - 0.5 + 1.5 * bias
    du_do   = where(cond, h, 0)

    bias = conf.surrogate_bias:  VGG16 0.6 / ResNet 0.8 / Spikformer 0.6

boxcar 는 폭 안에서 상수 `1/(2*width)` 를 주는데, asym 은 **막전위에 비례**하는 값을 준다.
bias=0.8 이면 h = 0.2*vmem + 0.7 — 문턱(vmem=1)에서 0.9, 아래쪽(vmem=0.2)에서 0.74 로
문턱에 가까울수록 크다. 이름 그대로 비대칭이다.

**옮길 때 주의한 두 가지**

1. SpikingJelly 는 surrogate 에 `v - v_threshold` 를 넘긴다 (`neuron.py` neuronal_fire).
   TF 는 raw `vmem` 을 쓰므로 `vmem = x + v_threshold` 로 되돌려야 한다.
   그래서 이 클래스는 **v_threshold 를 생성자로 받는다.**
   QKFormer 는 attn_lif 만 0.5 이고 나머지는 1.0 이라, 모듈마다 맞는 값을 줘야 한다.

2. QKFormer 는 `backend='cupy'` 라 파이썬 backward 가 아니라 `cuda_code` 를 쓴다.
   둘 다 구현하고 같은 식을 쓰게 했다 (`torch` 백엔드로도 검증 가능하도록).
"""
import torch
import torch.nn as nn
from spikingjelly.clock_driven.surrogate import (
    SurrogateFunctionBase, heaviside, tab4_str,
)


class _asym(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, bias, vth):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha, ctx.bias, ctx.vth = alpha, bias, vth
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # SpikingJelly 는 x = vmem - vth 를 넘기므로 되돌린다
        vmem = x + ctx.vth
        # TF 는 |vmem-vth|<=width 가 아니라 두 조건을 따로 본다
        # (`vmem >= vth-width` and `vmem <= vth+width`). 수학적으로는 같지만
        # 부동소수점에서 경계점 판정이 갈린다 (vth=1.0,width=0.5 일 때 vmem=0.5 에서
        # 실제로 갈렸다). 비트 단위로 맞추려고 TF 와 같은 형태로 쓴다.
        inside = ((vmem >= ctx.vth - ctx.alpha) & (vmem <= ctx.vth + ctx.alpha)).to(x)
        h = (1.0 - ctx.bias) * vmem - 0.5 + 1.5 * ctx.bias
        return grad_output * h * inside, None, None, None


class Asym(SurrogateFunctionBase):
    """TF 의 asym 대체 기울기.

    Args:
        alpha: 문턱 좌우 폭 (TF `surro_grad_alpha`, 기본 0.5)
        bias:  기울기 기울기 (TF `surrogate_bias`) — ResNet 0.8 / VGG16·Spikformer 0.6
        v_threshold: 그 뉴런의 문턱. SpikingJelly 가 x = v - vth 를 넘기므로 필요하다
    """

    def __init__(self, alpha=0.5, bias=0.6, v_threshold=1.0, spiking=True):
        super().__init__(alpha, spiking)
        self.bias = bias
        self.v_threshold = v_threshold

    def extra_repr(self):
        return (f'alpha={self.alpha}, bias={self.bias}, '
                f'v_threshold={self.v_threshold}, spiking={self.spiking}')

    def forward(self, x: torch.Tensor):
        if self.spiking:
            return _asym.apply(x, self.alpha, self.bias, self.v_threshold)
        # 비-스파이킹 모드(ANN 근사)는 이 실험에서 안 쓴다
        raise NotImplementedError('Asym 은 spiking 모드만 지원한다')

    def cuda_code(self, x: str, y: str, dtype='fp32'):
        """cupy 백엔드용. 파이썬 backward 와 같은 식이어야 한다."""
        sg = 'sg_' + self._get_name()
        a = str(self.alpha) + 'f'
        b = str(self.bias) + 'f'
        vth = str(self.v_threshold) + 'f'
        code = f'''
            {tab4_str}{self.cuda_code_start_comments()}
        '''
        if dtype == 'fp32':
            code += f'''
            {tab4_str}const float {sg}_vmem = {x} + {vth};
            {tab4_str}const float {sg}_h = (1.0f - {b}) * {sg}_vmem - 0.5f + 1.5f * {b};
            {tab4_str}const float {y} = ({sg}_vmem >= {vth} - {a} && {sg}_vmem <= {vth} + {a}) ? {sg}_h : 0.0f;
            '''
        elif dtype == 'fp16':
            code += f'''
            {tab4_str}const half2 {sg}_vth = __float2half2_rn({vth});
            {tab4_str}const half2 {sg}_b = __float2half2_rn({b});
            {tab4_str}const half2 {sg}_vmem = __hadd2({x}, {sg}_vth);
            {tab4_str}const half2 {sg}_h = __hadd2(__hmul2(__hsub2(__float2half2_rn(1.0f), {sg}_b), {sg}_vmem), __hsub2(__hmul2(__float2half2_rn(1.5f), {sg}_b), __float2half2_rn(0.5f)));
            {tab4_str}const half2 {sg}_a = __float2half2_rn({a});
            {tab4_str}const half2 {sg}_in = __hmul2(__hge2({sg}_vmem, __hsub2({sg}_vth, {sg}_a)), __hle2({sg}_vmem, __hadd2({sg}_vth, {sg}_a)));
            {tab4_str}const half2 {y} = __hmul2({sg}_h, {sg}_in);
            '''
        else:
            raise NotImplementedError
        code += f'''
            {tab4_str}{self.cuda_code_end_comments()}
        '''
        return code


# TF 쪽 config_snn_training.py:174-181 과 같은 표
BIAS_BY_MODEL = {'VGG16': 0.6, 'ResNet': 0.8, 'Spikformer': 0.6}


def apply_asym(model, alpha=0.5, bias=0.6, types=None, verbose=True):
    """모델의 모든 스파이킹 뉴런 대체 기울기를 asym 으로 바꾼다.

    각 뉴런의 `v_threshold` 를 읽어 그 값에 맞는 인스턴스를 따로 만든다 —
    QKFormer 는 attn_lif 만 0.5 라 전역 하나로는 그 층이 틀린다.
    """
    from spikingjelly.clock_driven.neuron import (
        MultiStepLIFNode, MultiStepParametricLIFNode,
    )
    types = types or (MultiStepLIFNode, MultiStepParametricLIFNode)
    n, vths = 0, {}
    for name, m in model.named_modules():
        if isinstance(m, types):
            vth = float(m.v_threshold)
            m.surrogate_function = Asym(alpha=alpha, bias=bias, v_threshold=vth)
            vths[vth] = vths.get(vth, 0) + 1
            n += 1
    if verbose:
        detail = ', '.join(f'vth={k}: {v}층' for k, v in sorted(vths.items()))
        print(f'[asym] 대체 기울기 교체 {n}개  (alpha={alpha}, bias={bias})  {detail}')
    return n
