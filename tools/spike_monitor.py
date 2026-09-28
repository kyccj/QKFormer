"""QKFormer 스파이크 발화량 측정.

모델 코드를 수정하지 않는다. QKFormer의 모든 스파이킹 뉴런이 명명된
모듈(`q_lif`, `k_lif`, `attn_lif`, `proj_lif`, `mlp1_lif`, ...)이므로
forward hook만으로 전 층의 출력 스파이크를 잡을 수 있다.

MultiStepLIFNode의 출력은 [T, B, ...] 모양의 {0,1} 텐서다. 따라서
sum이 곧 스파이크 수이고 mean이 발화율이다.

    from tools.spike_monitor import SpikeMonitor

    mon = SpikeMonitor(model)
    for x, y in loader:
        model(x); functional.reset_net(model)
        mon.step(batch_size=x.shape[0])
    print(mon.report())
    mon.remove()

주의: 측정만 한다. hook에서 텐서를 detach하므로 학습 그래프에 영향이 없고
메모리도 누적되지 않는다. 규제(EIP)로 쓰려면 detach를 빼야 하는데, 그 경우
cupy backend에서 backward가 의도대로 흐르는지 별도 검증이 필요하다.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.nn as nn

try:
    from spikingjelly.clock_driven.neuron import (
        MultiStepLIFNode,
        MultiStepParametricLIFNode,
    )
    SPIKING_TYPES = (MultiStepLIFNode, MultiStepParametricLIFNode)
except ImportError:  # pragma: no cover
    SPIKING_TYPES = ()


class SpikeMonitor:
    """스파이킹 뉴런 층별 발화량을 누적 집계한다."""

    def __init__(self, model: nn.Module, types=None, T: Optional[int] = None):
        self.types = types or SPIKING_TYPES
        if not self.types:
            raise RuntimeError("spikingjelly 뉴런 타입을 찾을 수 없다.")

        # 층 출력의 shape[0]을 T로 믿으면 안 된다. QKFormer의 SSA는 proj_lif
        # 직전에 x.flatten(0,1)로 T와 B를 합치므로 그 층은 shape[0]=T*B가 된다.
        # T는 모델에서 직접 받고, 뉴런 수는 numel/(T*batch)로 역산한다.
        self.T = T if T is not None else getattr(model, "T", None)
        if self.T is None:
            raise ValueError("T를 알 수 없다. model.T가 없으면 T= 로 넘길 것.")

        self._handles = []
        self._pending: Dict[str, torch.Tensor] = {}

        # 층별 누적치
        self.spikes: "OrderedDict[str, float]" = OrderedDict()   # 총 스파이크 수
        self.elements: "OrderedDict[str, float]" = OrderedDict() # 총 (뉴런 x 시점) 수
        self.neurons: "OrderedDict[str, int]" = OrderedDict()    # 샘플당 뉴런 수
        self.timesteps: "OrderedDict[str, int]" = OrderedDict()
        self.samples = 0

        for name, module in model.named_modules():
            if isinstance(module, self.types):
                self._handles.append(
                    module.register_forward_hook(self._make_hook(name))
                )
        if not self._handles:
            raise RuntimeError("스파이킹 뉴런 모듈을 하나도 찾지 못했다.")

    def _make_hook(self, name: str):
        def hook(_module, _inp, out):
            if isinstance(out, tuple):
                out = out[0]
            # [T, B, ...] {0,1}. detach — 측정이 학습에 개입하지 않도록.
            self._pending[name] = out.detach()
        return hook

    def step(self, batch_size: int) -> None:
        """한 배치의 forward가 끝난 뒤 호출한다."""
        if not self._pending:
            raise RuntimeError("hook이 잡은 출력이 없다. forward를 먼저 실행할 것.")
        for name, out in self._pending.items():
            n = out.numel()
            self.spikes[name] = self.spikes.get(name, 0.0) + out.sum().item()
            self.elements[name] = self.elements.get(name, 0.0) + n
            # 샘플 1개 · 시점 1개당 뉴런 수. shape에 의존하지 않는다.
            self.neurons[name] = n // (self.T * batch_size)
            self.timesteps[name] = self.T
        self.samples += batch_size
        self._pending.clear()

    # --- 결과 ---------------------------------------------------------

    def layer_rows(self) -> List[dict]:
        """층별 결과. spikes_per_sample은 샘플 1개를 추론할 때의 스파이크 수."""
        rows = []
        for name in self.spikes:
            total = self.spikes[name]
            rows.append({
                "layer": name,
                "firing_rate": total / self.elements[name],
                "spikes_per_sample": total / self.samples,
                "neurons": self.neurons[name],
                "T": self.timesteps[name],
            })
        return rows

    def totals(self) -> dict:
        total_spikes = sum(self.spikes.values())
        total_elements = sum(self.elements.values())
        return {
            "samples": self.samples,
            "layers": len(self.spikes),
            "total_neurons": sum(self.neurons.values()),
            "spikes_per_sample": total_spikes / self.samples,
            "firing_rate": total_spikes / total_elements,
        }

    def report(self, top: Optional[int] = None) -> str:
        rows = sorted(self.layer_rows(), key=lambda r: -r["spikes_per_sample"])
        if top:
            rows = rows[:top]
        t = self.totals()
        w = max(len(r["layer"]) for r in rows)
        lines = [
            f"{'layer':<{w}}  {'fire rate':>9}  {'spikes/sample':>14}  {'neurons':>9}  {'T':>2}",
            "-" * (w + 42),
        ]
        for r in rows:
            lines.append(
                f"{r['layer']:<{w}}  {r['firing_rate']:>9.4f}  "
                f"{r['spikes_per_sample']:>14,.1f}  {r['neurons']:>9,}  {r['T']:>2}"
            )
        lines += [
            "-" * (w + 42),
            f"{'TOTAL':<{w}}  {t['firing_rate']:>9.4f}  "
            f"{t['spikes_per_sample']:>14,.1f}  {t['total_neurons']:>9,}",
            f"({t['samples']:,} samples, {t['layers']} spiking layers)",
        ]
        return "\n".join(lines)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()
