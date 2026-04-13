"""
tokenizer/bpe_tokenizer.py — AIONTokenizer: tokenizador BPE de 32K tokens
===========================================================================

Entrena un tokenizador SentencePiece BPE sobre el corpus completo de
AION-C (6 archivos JSONL: cora, forge_c, axiom, muse, empathy, orchestrator)
más texto de cobertura de caracteres para alcanzar vocab_size=32K.

El corpus sintético tiene vocabulario limitado (~6K palabras únicas). Para
garantizar 32K tokens, BPETrainer añade automáticamente texto de cobertura:
combinaciones de caracteres del español+inglés, sufijos/prefijos morfológicos,
números, y frases cortas variadas. Esto es práctica estándar para tokenizadores
en corpora pequeños.

Uso — Entrenamiento (una sola vez):
    from tokenizer.bpe_tokenizer import BPETrainer
    trainer = BPETrainer(
        corpus_dir="/path/to/datasets",
        output_model="tokenizer/aion_32k.model",
    )
    trainer.train()

Uso — Inferencia:
    from tokenizer import AIONTokenizer
    tok = AIONTokenizer("tokenizer/aion_32k.model")
    ids = tok.encode("Hola mundo")           # → list[int]
    txt = tok.decode(ids)                    # → "Hola mundo"
    print(tok.vocab_size)                    # → 32000
    print(tok.pad_id, tok.bos_id, tok.eos_id)  # → 0, 1, 2

Tokens especiales (posiciones fijas):
    pad_id = 0   <pad>
    bos_id = 1   <s>
    eos_id = 2   </s>
    unk_id = 3   <unk>
"""

from __future__ import annotations

import io
import json
import os
import re
import tempfile
from pathlib import Path
from typing import List, Optional, Union

import sentencepiece as spm


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

VOCAB_SIZE       = 32_000
PAD_ID           = 0
BOS_ID           = 1
EOS_ID           = 2
UNK_ID           = 3

_CORPUS_FILES = [
    "mose_cora.jsonl",
    "mose_forge_c.jsonl",
    "mose_axiom.jsonl",
    "mose_muse.jsonl",
    "mose_empathy.jsonl",
    "mose_orchestrator.jsonl",
]

_TEXT_FIELDS = ["input", "expected_output", "reasoning"]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS INTERNOS
# ─────────────────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    """
    Divide un texto largo en oraciones por puntuación y saltos de línea.
    """
    parts = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [p.strip() for p in parts if p.strip()]


def _write_char_coverage(out: io.TextIOWrapper) -> int:
    """
    Escribe texto sistemático para garantizar que SentencePiece genere
    suficientes piezas subpalabra para alcanzar 32K tokens BPE.

    BPE necesita ~32K merges únicos. Para un corpus con ~6K palabras únicas,
    generamos combinaciones exhaustivas de caracteres como "palabras" de
    entrenamiento, dando a BPE todos los patrones que necesita.

    Con 50 caracteres base:
    - bigrams: 50^2 = 2,500 palabras únicas
    - trigrams: 50^3 = 125,000 palabras únicas
    - 4-grams de 20 chars: 20^4 = 160,000
    Total >> 32K merges posibles.
    """
    n = 0

    # Caracteres más comunes del español e inglés (ordenados por frecuencia)
    COMMON = list("aeiounsrlctdpmghbfvyáéíóúAEIOUNSRLCTDPMGHBFVYqjwxzkz")
    COMMON = list(dict.fromkeys(COMMON))  # deduplicar, mantener orden

    # Puntuación ASCII para garantizar que sea parte del vocab
    PUNCT_CHARS = list("!@#$%^&*;:~`|\\<>[]{}+=_?/,'\"")

    # Top 50 caracteres para cobertura
    C50 = COMMON[:50]
    # Top 25 para 4-grams (25^4 = 390,625 combinaciones)
    C25 = COMMON[:25]
    # Top 15 para 5-grams
    C15 = COMMON[:15]

    # 1. Todos los bigrams: ~2,500 palabras
    for c1 in C50:
        for c2 in C50:
            out.write(c1 + c2 + "\n")
            n += 1

    # 2. Todos los trigrams de C50: ~125,000 palabras
    line_buf: List[str] = []
    for c1 in C50:
        for c2 in C50:
            for c3 in C50:
                line_buf.append(c1 + c2 + c3)
                if len(line_buf) >= 10:
                    out.write(" ".join(line_buf) + "\n")
                    n += 1
                    line_buf = []
    if line_buf:
        out.write(" ".join(line_buf) + "\n")
        n += 1

    # 3. 4-grams de C25: 390,625 combinaciones únicas
    line_buf = []
    for c1 in C25:
        for c2 in C25:
            for c3 in C25:
                for c4 in C25:
                    line_buf.append(c1 + c2 + c3 + c4)
                    if len(line_buf) >= 8:
                        out.write(" ".join(line_buf) + "\n")
                        n += 1
                        line_buf = []
    if line_buf:
        out.write(" ".join(line_buf) + "\n")
        n += 1

    # 4. 5-grams de C15 para más profundidad de merges
    line_buf = []
    for c1 in C15:
        for c2 in C15:
            for c3 in C15:
                for c4 in C15:
                    for c5 in C15:
                        line_buf.append(c1 + c2 + c3 + c4 + c5)
                        if len(line_buf) >= 6:
                            out.write(" ".join(line_buf) + "\n")
                            n += 1
                            line_buf = []
    if line_buf:
        out.write(" ".join(line_buf) + "\n")
        n += 1

    # 5. Caracteres de puntuación en contexto (para que participen en merges BPE)
    for p in PUNCT_CHARS:
        for c in C25[:15]:
            # Contextos: c+p, p+c, c+p+c, p+c+p
            out.write(f"{c}{p} {p}{c} {c}{p}{c} {p}{c}{p}\n")
            n += 1
        # Trigrams con puntuación
        for c1 in C25[:10]:
            for c2 in C25[:10]:
                out.write(f"{c1}{p}{c2} {p}{c1}{c2} {c1}{c2}{p}\n")
                n += 1

    # 6. Números 0-9999 para cobertura numérica
    for i in range(10000):
        out.write(str(i) + "\n")
        n += 1

    return n


# ─────────────────────────────────────────────────────────────────────────────
# BPETrainer — extrae texto del corpus y entrena SentencePiece
# ─────────────────────────────────────────────────────────────────────────────

class BPETrainer:
    """
    Entrena un tokenizador BPE de 32K tokens sobre el corpus de AION-C.

    El corpus se lee de los 6 archivos JSONL, extrayendo los campos:
        input, expected_output, reasoning

    Args:
        corpus_dir:   Directorio con los archivos JSONL del corpus.
        output_model: Ruta donde guardar el modelo .model de SentencePiece.
        vocab_size:   Tamaño del vocabulario. Por defecto 32_000.
        character_coverage: Cobertura de caracteres. 0.9995 cubre español+inglés.
        input_sentence_size: Máximo de frases para el entrenamiento (0 = sin límite).
        num_threads:  Threads para SentencePiece. Default 4.
    """

    def __init__(
        self,
        corpus_dir:            str,
        output_model:          str  = "tokenizer/aion_32k.model",
        vocab_size:            int  = VOCAB_SIZE,
        character_coverage:    float = 0.9995,
        input_sentence_size:   int  = 0,
        num_threads:           int  = 4,
    ) -> None:
        self.corpus_dir          = Path(corpus_dir)
        self.output_model        = Path(output_model)
        self.vocab_size          = vocab_size
        self.character_coverage  = character_coverage
        self.input_sentence_size = input_sentence_size
        self.num_threads         = num_threads

    # ── API principal ─────────────────────────────────────────────────────────

    def train(self, verbose: bool = True) -> None:
        """
        Extrae texto del corpus y entrena el modelo BPE.

        El modelo se guarda en self.output_model.
        """
        self.output_model.parent.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"[BPETrainer] Extrayendo texto de {str(self.corpus_dir)} ...")

        # Escribir texto en archivo temporal para SentencePiece
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", encoding="utf-8", delete=False
        ) as tmp:
            tmp_path = tmp.name
            n_sentences = self._write_corpus(tmp, verbose=verbose)

        if verbose:
            print(f"[BPETrainer] {n_sentences:,} frases extraidas -> {tmp_path}")
            print(f"[BPETrainer] Entrenando BPE vocab_size={self.vocab_size} ...")

        try:
            self._run_training(tmp_path)
        finally:
            os.unlink(tmp_path)

        if verbose:
            print(f"[BPETrainer] Modelo guardado en {self.output_model}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _write_corpus(self, out: io.TextIOWrapper, verbose: bool = False) -> int:
        """
        Lee los 6 JSONL y escribe una línea por frase en `out`.

        También escribe texto de cobertura de caracteres para asegurar que
        SentencePiece pueda generar suficientes piezas subpalabra para 32K.

        Retorna el número de frases escritas.
        """
        n = 0

        # ── 1. Texto real del corpus ──────────────────────────────────────────
        for filename in _CORPUS_FILES:
            path = self.corpus_dir / filename
            if not path.exists():
                if verbose:
                    print(f"  [!] {filename} no encontrado, omitiendo")
                continue

            if verbose:
                size_mb = path.stat().st_size / 1_048_576
                print(f"  Leyendo {filename} ({size_mb:.1f} MB) ...")

            with open(path, encoding="utf-8") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        record = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    for field in _TEXT_FIELDS:
                        text = record.get(field, "")
                        if not isinstance(text, str):
                            text = str(text) if text else ""
                        text = text.strip()
                        if text:
                            # Escribir el texto completo
                            out.write(text + "\n")
                            n += 1
                            # También dividir en oraciones para más diversidad
                            for sent in _split_sentences(text):
                                if len(sent) > 10:
                                    out.write(sent + "\n")
                                    n += 1

        # ── 2. Texto de cobertura de caracteres ───────────────────────────────
        # El corpus sintético tiene vocabulario limitado (~6K palabras únicas).
        # Para llegar a 32K tokens, añadimos texto que cubra todas las
        # combinaciones de caracteres del español e inglés.
        n += _write_char_coverage(out)
        if verbose:
            print("  Cobertura de caracteres escrita.")

        return n

    def _run_training(self, input_path: str) -> None:
        """
        Llama a sentencepiece.SentencePieceTrainer.train con los parámetros
        correctos para reproducir tokens especiales en posiciones fijas.

        Tokens especiales fijos:
            pos 0 → <pad>   (pad)
            pos 1 → <s>     (bos)
            pos 2 → </s>    (eos)
            pos 3 → <unk>   (unk, posición estándar SP)
        """
        model_prefix = str(self.output_model.with_suffix(""))

        # SentencePiece pone <unk>=0, <s>=1, </s>=2 por defecto.
        # Para mover pad a 0, usamos control_symbols.
        # Usamos user_defined_symbols para <pad> y lo ponemos primero.
        train_args = dict(
            input                     = input_path,
            model_prefix              = model_prefix,
            vocab_size                = self.vocab_size,
            model_type                = "bpe",
            # character_coverage=1.0 garantiza que TODOS los caracteres que
            # aparecen en el corpus (incluyendo ! ; @ # etc.) sean tokens.
            character_coverage        = 1.0,
            pad_id                    = PAD_ID,
            bos_id                    = BOS_ID,
            eos_id                    = EOS_ID,
            unk_id                    = UNK_ID,
            pad_piece                 = "<pad>",
            bos_piece                 = "<s>",
            eos_piece                 = "</s>",
            unk_piece                 = "<unk>",
            num_threads               = self.num_threads,
            shuffle_input_sentence    = True,
            # No truncar textos largos
            max_sentence_length       = 65536,
            split_digits              = True,
        )
        if self.input_sentence_size > 0:
            train_args["input_sentence_size"]        = self.input_sentence_size
            train_args["shuffle_input_sentence"]     = True

        spm.SentencePieceTrainer.train(**train_args)


# ─────────────────────────────────────────────────────────────────────────────
# AIONTokenizer — wrapper sobre el modelo entrenado
# ─────────────────────────────────────────────────────────────────────────────

class AIONTokenizer:
    """
    Tokenizador BPE de 32K tokens para AION-C.

    Wrappea un modelo SentencePiece entrenado con BPETrainer.

    Uso:
        tok = AIONTokenizer("tokenizer/aion_32k.model")
        ids = tok.encode("Hola mundo")
        txt = tok.decode(ids)
        assert txt == "Hola mundo"

    Args:
        model_path: Ruta al archivo .model de SentencePiece.
    """

    def __init__(self, model_path: Union[str, Path]) -> None:
        self._model_path = Path(model_path)
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Modelo SentencePiece no encontrado: {self._model_path}\n"
                f"Entrena primero con BPETrainer.train()"
            )
        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(str(self._model_path))

        # Verificar que los IDs especiales coincidan con las constantes
        assert self._sp.pad_id()  == PAD_ID,  f"pad_id mismatch: {self._sp.pad_id()} != {PAD_ID}"
        assert self._sp.bos_id()  == BOS_ID,  f"bos_id mismatch: {self._sp.bos_id()} != {BOS_ID}"
        assert self._sp.eos_id()  == EOS_ID,  f"eos_id mismatch: {self._sp.eos_id()} != {EOS_ID}"
        assert self._sp.unk_id()  == UNK_ID,  f"unk_id mismatch: {self._sp.unk_id()} != {UNK_ID}"

    # ── Propiedades ───────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        """Tamaño del vocabulario (32_000)."""
        return self._sp.GetPieceSize()

    @property
    def pad_id(self) -> int:
        """ID del token de padding (<pad> = 0)."""
        return PAD_ID

    @property
    def bos_id(self) -> int:
        """ID del token de inicio de secuencia (<s> = 1)."""
        return BOS_ID

    @property
    def eos_id(self) -> int:
        """ID del token de fin de secuencia (</s> = 2)."""
        return EOS_ID

    @property
    def unk_id(self) -> int:
        """ID del token desconocido (<unk> = 3)."""
        return UNK_ID

    @property
    def model_path(self) -> Path:
        """Ruta al archivo .model cargado."""
        return self._model_path

    # ── API principal ─────────────────────────────────────────────────────────

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """
        Codifica texto a lista de IDs enteros.

        Args:
            text:    Texto a tokenizar.
            add_bos: Si True, antepone bos_id.
            add_eos: Si True, agrega eos_id al final.

        Returns:
            Lista de enteros en [0, vocab_size).
        """
        ids: List[int] = self._sp.EncodeAsIds(text)
        if add_bos:
            ids = [BOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]
        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Decodifica lista de IDs a string.

        Filtra automáticamente los tokens especiales (pad, bos, eos).

        Args:
            ids: Lista de enteros.

        Returns:
            Texto reconstruido.
        """
        # Filtrar tokens especiales antes de decodificar
        filtered = [i for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID)]
        return self._sp.DecodeIds(filtered)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokeniza texto a lista de strings (piezas BPE).

        Útil para inspección y debugging.
        """
        return self._sp.EncodeAsPieces(text)

    def piece_to_id(self, piece: str) -> int:
        """Convierte una pieza BPE a su ID."""
        return self._sp.PieceToId(piece)

    def id_to_piece(self, id: int) -> str:
        """Convierte un ID a su pieza BPE."""
        return self._sp.IdToPiece(id)

    def encode_batch(self, texts: List[str], add_bos: bool = False, add_eos: bool = False) -> List[List[int]]:
        """Codifica una lista de textos en batch."""
        return [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]

    def decode_batch(self, batch: List[List[int]]) -> List[str]:
        """Decodifica un batch de listas de IDs."""
        return [self.decode(ids) for ids in batch]

    # ── Utilidades ────────────────────────────────────────────────────────────

    def coverage(self, text: str) -> float:
        """
        Fracción de tokens en el texto que NO son <unk>.

        Un valor cercano a 1.0 indica buena cobertura del vocabulario.

        Args:
            text: Texto de prueba.

        Returns:
            Float en [0.0, 1.0].
        """
        ids = self.encode(text)
        if not ids:
            return 1.0
        n_unk = sum(1 for i in ids if i == UNK_ID)
        return 1.0 - n_unk / len(ids)

    def __repr__(self) -> str:
        return (
            f"AIONTokenizer("
            f"vocab_size={self.vocab_size}, "
            f"model={self._model_path.name!r}"
            f")"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI helper — permite entrenar desde la línea de comandos
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Entrenador BPE de AION-C")
    parser.add_argument(
        "--corpus-dir",
        required=True,
        help="Directorio con los archivos JSONL del corpus",
    )
    parser.add_argument(
        "--output",
        default="tokenizer/aion_32k.model",
        help="Ruta de salida para el modelo .model (default: tokenizer/aion_32k.model)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=VOCAB_SIZE,
        help=f"Tamaño del vocabulario (default: {VOCAB_SIZE})",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=0,
        help="Máximo de frases para entrenamiento (0 = sin límite)",
    )
    args = parser.parse_args()

    trainer = BPETrainer(
        corpus_dir          = args.corpus_dir,
        output_model        = args.output,
        vocab_size          = args.vocab_size,
        input_sentence_size = args.max_sentences,
    )
    trainer.train(verbose=True)
    print("Listo.")
