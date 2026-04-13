"""
inference/quantize.py — Cuantización post-entrenamiento para AION-C
===================================================================

Toma el checkpoint final y produce una versión INT4 cuantizada.
Guarda como aion_c_3.5b_int4.safetensors (o .pt si safetensors no disponible).

Estrategia: cuantización por canal (per-channel) a INT4 con scale+zero_point.
Si bitsandbytes está disponible, usa NF4. Sino, usa cuantización naive por grupo.

Uso:
    python -m inference.quantize --checkpoint runs/aion_production/phase4_instruction.pt
    python -m inference.quantize --checkpoint runs/aion_tiny/phase4_instruction.pt --output model_int4.pt
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

import sys, os
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# NAIVE INT4 QUANTIZATION (fallback cuando no hay bitsandbytes)
# ─────────────────────────────────────────────────────────────────────────────

def quantize_tensor_int4(tensor: torch.Tensor, group_size: int = 128) -> Dict:
    """
    Cuantiza un tensor a INT4 naive con scale por grupo.

    Args:
        tensor: Tensor float a cuantizar.
        group_size: Tamaño de grupo para scale (default 128).

    Returns:
        dict con: quantized (uint8, packed), scales, zeros, shape, group_size
    """
    original_shape = tensor.shape
    flat = tensor.flatten().float()
    n = flat.numel()

    # Pad to group_size
    if n % group_size != 0:
        pad_size = group_size - (n % group_size)
        flat = torch.cat([flat, torch.zeros(pad_size)])
    else:
        pad_size = 0

    n_groups = flat.numel() // group_size
    flat = flat.view(n_groups, group_size)

    # Per-group quantization
    mins = flat.min(dim=1, keepdim=True).values
    maxs = flat.max(dim=1, keepdim=True).values
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0  # avoid division by zero

    scales = ranges / 15.0  # INT4 → 0-15
    zeros  = mins

    # Quantize to [0, 15]
    quantized = ((flat - zeros) / scales).round().clamp(0, 15).to(torch.uint8)

    # Pack two INT4 values per uint8
    quantized_flat = quantized.flatten()
    n_packed = quantized_flat.numel()
    if n_packed % 2 != 0:
        quantized_flat = torch.cat([quantized_flat, torch.zeros(1, dtype=torch.uint8)])
    packed = (quantized_flat[0::2] << 4) | quantized_flat[1::2]

    return {
        "packed":     packed,
        "scales":     scales.squeeze(1),
        "zeros":      zeros.squeeze(1),
        "shape":      list(original_shape),
        "group_size": group_size,
        "pad_size":   pad_size,
    }


def dequantize_tensor_int4(qdata: Dict) -> torch.Tensor:
    """Dequantiza un tensor INT4 a float."""
    packed     = qdata["packed"]
    scales     = qdata["scales"]
    zeros      = qdata["zeros"]
    shape      = qdata["shape"]
    group_size = qdata["group_size"]
    pad_size   = qdata.get("pad_size", 0)

    # Unpack
    high = (packed >> 4) & 0x0F
    low  = packed & 0x0F
    unpacked = torch.stack([high, low], dim=1).flatten().float()

    # Remove padding
    n_total = 1
    for s in shape:
        n_total *= s
    total_groups = (n_total + pad_size) // group_size
    unpacked = unpacked[:total_groups * group_size]

    # Reshape to groups
    unpacked = unpacked.view(total_groups, group_size)

    # Dequantize
    result = unpacked * scales.unsqueeze(1) + zeros.unsqueeze(1)

    # Flatten and trim
    result = result.flatten()[:n_total]
    return result.view(shape)


# ─────────────────────────────────────────────────────────────────────────────
# QUANTIZE MODEL
# ─────────────────────────────────────────────────────────────────────────────

def quantize_state_dict(state_dict: Dict, group_size: int = 128) -> Dict:
    """
    Cuantiza todos los tensores float del state_dict a INT4.

    Tensores de 1 dimensión (biases, norms) se mantienen en float.
    """
    quantized = {}
    stats = {"quantized": 0, "kept_float": 0, "total_params": 0}

    for key, tensor in state_dict.items():
        stats["total_params"] += tensor.numel()
        if tensor.ndim >= 2 and tensor.numel() >= group_size:
            qdata = quantize_tensor_int4(tensor, group_size)
            quantized[key] = {"__int4__": True, **{k: v for k, v in qdata.items()}}
            stats["quantized"] += tensor.numel()
        else:
            quantized[key] = tensor
            stats["kept_float"] += tensor.numel()

    return quantized, stats


def dequantize_state_dict(quantized: Dict) -> Dict:
    """Dequantiza un state_dict INT4 a float."""
    state_dict = {}
    for key, val in quantized.items():
        if isinstance(val, dict) and val.get("__int4__"):
            qdata = {k: v for k, v in val.items() if k != "__int4__"}
            state_dict[key] = dequantize_tensor_int4(qdata)
        else:
            state_dict[key] = val
    return state_dict


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Quantize AION-C checkpoint to INT4")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", default=None, help="Output path (default: same dir with _int4 suffix)")
    parser.add_argument("--group-size", type=int, default=128, help="Group size for quantization")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found: {ckpt_path}")
        return

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state", ckpt)

    print(f"Quantizing to INT4 (group_size={args.group_size})...")
    quantized, stats = quantize_state_dict(state_dict, args.group_size)

    q_ratio = stats["quantized"] / max(1, stats["total_params"])
    print(f"  Total params:    {stats['total_params']:,}")
    print(f"  Quantized:       {stats['quantized']:,} ({q_ratio:.1%})")
    print(f"  Kept float:      {stats['kept_float']:,}")

    # Output path
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = ckpt_path.parent / f"{ckpt_path.stem}_int4.pt"

    # Try safetensors first, fallback to torch
    try:
        from safetensors.torch import save_file
        # safetensors requires flat tensors, so we save as .pt for complex dicts
        raise ImportError("Complex dict not supported by safetensors")
    except ImportError:
        torch.save({"quantized_state": quantized, "config": ckpt.get("config")}, out_path)
        print(f"  Saved: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
