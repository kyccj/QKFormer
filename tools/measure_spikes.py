"""학습된 QKFormer 체크포인트의 스파이크 발화량을 측정한다.

학습 때 저장된 args.yaml을 그대로 읽어 모델을 재구성하므로, 인자를 손으로
맞출 필요가 없다.

    python tools/measure_spikes.py \
        --run-dir ~/runs/qkformer_baseline/c10_baseline \
        --checkpoint model_best.pth.tar \
        --data-dir ~/.keras/datasets \
        --limit 2000 \
        --out spikes_c10.json

정확도도 함께 출력한다. 체크포인트를 제대로 불러왔는지 확인하는 용도로,
학습 로그의 best와 일치해야 한다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def build_model(args_yaml: dict):
    """train.py의 create_model 호출을 그대로 재현한다."""
    from timm.models import create_model

    ds = args_yaml.get("dataset", "")
    variant = "cifar100" if "cifar100" in ds else "cifar10"
    sys.path.insert(0, str(REPO / variant))
    import model as _model_module  # noqa: F401  (@register_model 등록용)

    return create_model(
        "QKFormer",
        pretrained=False,
        drop_rate=0.0,
        drop_path_rate=0.2,
        drop_block_rate=None,
        img_size_h=args_yaml["img_size"], img_size_w=args_yaml["img_size"],
        patch_size=args_yaml["patch_size"],
        embed_dims=args_yaml["dim"],
        num_heads=args_yaml["num_heads"],
        mlp_ratios=args_yaml["mlp_ratio"],
        in_channels=3,
        num_classes=args_yaml["num_classes"],
        qkv_bias=False,
        depths=args_yaml["layer"],
        sr_ratios=1,
        T=args_yaml["time_step"],
    ), variant


def build_loader(args_yaml: dict, data_dir: str, batch_size: int):
    from timm.data import create_dataset, create_loader, resolve_data_config

    dataset = create_dataset(
        args_yaml["dataset"], root=data_dir,
        split=args_yaml.get("val_split", "validation"),
        is_training=False, batch_size=batch_size,
    )
    cfg = resolve_data_config(args_yaml, model=None, verbose=False)
    cfg["input_size"] = (3, args_yaml["img_size"], args_yaml["img_size"])
    cfg["mean"] = tuple(args_yaml["mean"])
    cfg["std"] = tuple(args_yaml["std"])
    cfg["crop_pct"] = args_yaml.get("crop_pct", 1.0)
    cfg["interpolation"] = args_yaml.get("interpolation", "bicubic")
    return create_loader(
        dataset, input_size=cfg["input_size"], batch_size=batch_size,
        is_training=False, use_prefetcher=True,
        interpolation=cfg["interpolation"], mean=cfg["mean"], std=cfg["std"],
        num_workers=4, crop_pct=cfg["crop_pct"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="args.yaml과 체크포인트가 있는 학습 출력 디렉토리")
    ap.add_argument("--checkpoint", default="model_best.pth.tar")
    ap.add_argument("--data-dir", default=None, help="생략 시 args.yaml 값 사용")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0,
                    help="평가할 최대 샘플 수 (0=전체)")
    ap.add_argument("--out", default=None, help="결과 JSON 경로")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    run_dir = Path(os.path.expanduser(args.run_dir))
    cfg = yaml.safe_load((run_dir / "args.yaml").read_text())
    data_dir = os.path.expanduser(args.data_dir or cfg["data_dir"])

    model, variant = build_model(cfg)
    ckpt_path = run_dir / args.checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] missing={len(missing)} unexpected={len(unexpected)}")
    print(f"체크포인트: {ckpt_path.name}"
          + (f"  (기록된 metric {ckpt['metric']:.2f})" if "metric" in ckpt else ""))

    model = model.to(args.device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"모델: {variant}, params {n_params:,}, T={cfg['time_step']}")

    loader = build_loader(cfg, data_dir, args.batch_size)

    from spikingjelly.clock_driven import functional
    from tools.spike_monitor import SpikeMonitor

    correct = seen = 0
    with SpikeMonitor(model) as mon, torch.no_grad():
        for x, y in loader:
            x, y = x.to(args.device), y.to(args.device)
            out = model(x)
            functional.reset_net(model)
            mon.step(batch_size=x.shape[0])
            correct += (out.argmax(1) == y).sum().item()
            seen += y.numel()
            if args.limit and seen >= args.limit:
                break

    acc = 100.0 * correct / seen
    print(f"\n정확도 {acc:.2f}%  ({seen:,} samples)\n")
    print(mon.report())

    if args.out:
        payload = {
            "variant": variant,
            "checkpoint": str(ckpt_path),
            "params": n_params,
            "T": cfg["time_step"],
            "accuracy": acc,
            "samples": seen,
            "totals": mon.totals(),
            "layers": mon.layer_rows(),
        }
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
