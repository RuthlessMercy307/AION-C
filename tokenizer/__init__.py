"""
tokenizer/ — Módulo de tokenización de AION-C
==============================================

Componentes:
  AIONTokenizer   (bpe_tokenizer.py) — wrapper SentencePiece BPE 32K
  BPETrainer      (bpe_tokenizer.py) — entrena el modelo sobre el corpus

Uso rápido:
    from tokenizer import AIONTokenizer
    tok = AIONTokenizer("tokenizer/aion_32k.model")
    ids = tok.encode("Hello world")
    txt = tok.decode(ids)
    assert txt == "Hello world"
"""

from .bpe_tokenizer import AIONTokenizer, BPETrainer

__all__ = ["AIONTokenizer", "BPETrainer"]
