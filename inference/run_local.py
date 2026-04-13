"""
inference/run_local.py -- Chat interactivo local con AION-C
==========================================================

Carga el modelo (cuantizado o no) y abre un chat en terminal.
Detecta automaticamente si hay CUDA disponible.
Auto-detecta el ultimo checkpoint en output/ si no se especifica.

Uso:
    # Auto-detectar checkpoint:
    python -m inference.run_local

    # Modelo especifico:
    python -m inference.run_local --checkpoint output/production/phase4_instruction.pt

    # Modelo cuantizado INT4:
    python -m inference.run_local --checkpoint output/production/production_int4.pt --quantized

    # Con system prompt:
    python -m inference.run_local --system "Eres un tutor de matematicas"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import torch

import sys, os
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


SYSTEM_PROMPT_DEFAULT = (
    "Eres AION-C, un asistente de IA creado por Jesus. "
    "Usas la arquitectura MoSE con 5 motores especializados. "
    "Responde de forma clara, directa y sin emojis. "
    "Si no sabes algo, admitelo honestamente."
)


def find_best_checkpoint() -> Optional[str]:
    """Auto-detect the best checkpoint in output/ directory."""
    output = _ROOT / "output"
    if not output.exists():
        return None
    # Prefer production > medium > tiny, prefer phase4 > phase3
    for config in ["production", "medium", "tiny"]:
        cfg_dir = output / config
        if not cfg_dir.exists():
            continue
        for name in ["phase4_instruction.pt", "phase3_final.pt",
                     f"{config}_int4.pt"]:
            p = cfg_dir / name
            if p.exists():
                return str(p)
    # Fallback: any .pt in output/
    pts = list(output.rglob("*.pt"))
    if pts:
        return str(max(pts, key=lambda p: p.stat().st_mtime))
    return None


def load_model(
    checkpoint_path: str,
    quantized: bool = False,
    device: Optional[str] = None,
):
    """
    Carga el pipeline AION-C desde un checkpoint.

    Returns:
        (pipeline, tok, device, mose_cfg)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    print(f"  Device: {device}")
    print(f"  Loading: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if quantized or "quantized_state" in ckpt:
        from inference.quantize import dequantize_state_dict
        print("  Dequantizing INT4 -> float32...")
        state_dict = dequantize_state_dict(ckpt["quantized_state"])
    else:
        state_dict = ckpt.get("model_state", ckpt)

    from router.pipeline import MoSEPipeline, MoSEConfig
    from experiments.train_production import build_tokenizer

    # Restore config
    config = ckpt.get("config", None)
    config_name = ckpt.get("config_name", None)

    if config is not None and isinstance(config, MoSEConfig):
        mose_cfg = config
    elif config_name == "production":
        mose_cfg = MoSEConfig(
            hidden_dim=1536, vocab_size=32_000,
            enc_n_layers=14, dec_n_layers=28,
            enc_state_dim=16, dec_state_dim=16,
            dec_n_heads=16, motor_n_heads=8, unif_n_heads=8,
            dec_max_seq_len=2048, orch_mlp_hidden=2048,
            motor_max_nodes=32,
        )
    elif config_name == "medium":
        mose_cfg = MoSEConfig(
            hidden_dim=768, vocab_size=32_000,
            enc_n_layers=6, dec_n_layers=12,
            enc_state_dim=16, dec_state_dim=16,
            dec_n_heads=8, motor_n_heads=8, unif_n_heads=8,
            dec_max_seq_len=512, orch_mlp_hidden=512,
        )
    else:
        print("  [INFO] Config not found in checkpoint, using tiny")
        mose_cfg = MoSEConfig.tiny()

    pipeline = MoSEPipeline(mose_cfg)

    try:
        pipeline.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"  [WARNING] Partial load: {e}")

    pipeline.to(device).eval()
    tok = build_tokenizer(mose_cfg.vocab_size)

    params = sum(p.numel() for p in pipeline.parameters())
    print(f"  Model loaded: {params:,} params ({params/1e6:.1f}M), vocab={tok.vocab_size}")

    return pipeline, tok, device, mose_cfg


def greedy_decode(
    pipeline,
    tok,
    text: str,
    device: torch.device,
    max_len: int = 128,
    system_prompt: str = "",
) -> str:
    """Genera respuesta con greedy decoding."""
    full_text = text
    if system_prompt:
        full_text = system_prompt + "\n" + text

    try:
        ids = tok.encode(full_text, max_len)
    except TypeError:
        ids = tok.encode(full_text)[:max_len]

    ids_t = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out = pipeline(ids_t)

    pred_ids = out.logits[0].argmax(dim=-1).tolist()

    try:
        return tok.decode(pred_ids)
    except Exception:
        return str(pred_ids[:30])


def chat_loop(pipeline, tok, device, system_prompt: str = "", max_len: int = 128):
    """Bucle de chat interactivo en terminal."""
    print("\n" + "=" * 60)
    print("  AION-C Interactive Chat")
    print("  Type 'quit' or 'exit' to leave")
    if system_prompt:
        print(f"  System: {system_prompt[:80]}")
    print("=" * 60 + "\n")

    history: List[str] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ("quit", "exit", ""):
            print("Goodbye!")
            break

        history.append(user_input)
        context = "\n".join(history[-6:])
        response = greedy_decode(pipeline, tok, context, device, max_len, system_prompt)

        print(f"AION-C: {response}\n")
        history.append(response)


def main():
    parser = argparse.ArgumentParser(description="AION-C Local Chat")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint (auto-detect if omitted)")
    parser.add_argument("--quantized", action="store_true", help="Checkpoint is INT4 quantized")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"],
                       help="Device (default: auto-detect)")
    parser.add_argument("--system", default=SYSTEM_PROMPT_DEFAULT, help="System prompt")
    parser.add_argument("--max-len", type=int, default=128, help="Max sequence length")
    args = parser.parse_args()

    ckpt = args.checkpoint
    if ckpt is None:
        ckpt = find_best_checkpoint()
        if ckpt is None:
            print("ERROR: No checkpoint found. Run training first or pass --checkpoint.")
            print("Searched in: output/")
            return
        print(f"Auto-detected checkpoint: {ckpt}")

    pipeline, tok, device, _ = load_model(ckpt, args.quantized, args.device)
    chat_loop(pipeline, tok, device, args.system, args.max_len)


if __name__ == "__main__":
    main()
