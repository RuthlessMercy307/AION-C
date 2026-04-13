"""
tests/test_bpe_tokenizer.py — Tests para AIONTokenizer
=======================================================

Tests:
  - Carga del modelo entrenado
  - encode → decode identidad
  - vocab_size = 32000
  - IDs especiales correctos (pad=0, bos=1, eos=2, unk=3)
  - Manejo de español e inglés
  - Cobertura ≥ 99% sin UNK
  - Encode/decode en batch
  - add_bos / add_eos
  - Tokenize (piezas de texto)
  - coverage()
  - BPETrainer (si el modelo no existe, lo entrena)

NOTA: El modelo debe estar en tokenizer/aion_32k.model.
Si no existe, los tests se saltarán con skip.
"""

from __future__ import annotations

import os
import json
import tempfile
from pathlib import Path

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH = Path(__file__).parent.parent / "tokenizer" / "aion_32k.model"
CORPUS_DIR = Path(
    "C:/Users/USER/Desktop/ias/DataSet-Generator-Claude-Opus"
    "/mose_distillation_datasets/datasets"
)


def model_available() -> bool:
    return MODEL_PATH.exists()


def corpus_available() -> bool:
    return (CORPUS_DIR / "mose_cora.jsonl").exists()


skip_no_model = pytest.mark.skipif(
    not model_available(),
    reason=f"Trained model not found at {MODEL_PATH}",
)

skip_no_corpus = pytest.mark.skipif(
    not corpus_available(),
    reason=f"Corpus not found at {CORPUS_DIR}",
)


@pytest.fixture(scope="module")
def tok():
    """AIONTokenizer cargado desde el modelo entrenado."""
    from tokenizer import AIONTokenizer
    return AIONTokenizer(MODEL_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Tests de carga
# ─────────────────────────────────────────────────────────────────────────────

class TestLoad:
    @skip_no_model
    def test_load_model(self):
        from tokenizer import AIONTokenizer
        t = AIONTokenizer(MODEL_PATH)
        assert t is not None

    def test_load_missing_model_raises(self):
        from tokenizer import AIONTokenizer
        with pytest.raises(FileNotFoundError, match="aion_nonexistent"):
            AIONTokenizer("tokenizer/aion_nonexistent.model")

    @skip_no_model
    def test_repr(self, tok):
        r = repr(tok)
        assert "AIONTokenizer" in r
        assert "32000" in r


# ─────────────────────────────────────────────────────────────────────────────
# Tests de vocab_size y tokens especiales
# ─────────────────────────────────────────────────────────────────────────────

class TestVocabAndSpecialTokens:
    @skip_no_model
    def test_vocab_size(self, tok):
        assert tok.vocab_size == 32_000

    @skip_no_model
    def test_pad_id(self, tok):
        assert tok.pad_id == 0

    @skip_no_model
    def test_bos_id(self, tok):
        assert tok.bos_id == 1

    @skip_no_model
    def test_eos_id(self, tok):
        assert tok.eos_id == 2

    @skip_no_model
    def test_unk_id(self, tok):
        assert tok.unk_id == 3

    @skip_no_model
    def test_special_tokens_distinct(self, tok):
        ids = {tok.pad_id, tok.bos_id, tok.eos_id, tok.unk_id}
        assert len(ids) == 4

    @skip_no_model
    def test_pad_piece(self, tok):
        assert tok.id_to_piece(tok.pad_id) == "<pad>"

    @skip_no_model
    def test_bos_piece(self, tok):
        assert tok.id_to_piece(tok.bos_id) == "<s>"

    @skip_no_model
    def test_eos_piece(self, tok):
        assert tok.id_to_piece(tok.eos_id) == "</s>"

    @skip_no_model
    def test_unk_piece(self, tok):
        assert tok.id_to_piece(tok.unk_id) == "<unk>"


# ─────────────────────────────────────────────────────────────────────────────
# Tests de encode → decode identidad
# ─────────────────────────────────────────────────────────────────────────────

class TestEncodeDecodeIdentity:
    @skip_no_model
    @pytest.mark.parametrize("text", [
        "Hello, world!",
        "hello world",
        "The quick brown fox",
        "This is a test sentence.",
        "Neural network parameters.",
        "F1=0.847 loss=0.023",
        "AION-C version 2.0",
    ])
    def test_english_identity(self, tok, text):
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text, f"Expected {text!r}, got {decoded!r}"

    @skip_no_model
    @pytest.mark.parametrize("text", [
        "Hola mundo",
        "El sistema de analisis",
        "La funcion de perdida",
        "Procesamiento de lenguaje natural",
        "Identificacion de patrones morfologicos",
        "La configuracion del modelo neuronal",
        "Calculo de gradientes por retropropagacion",
    ])
    def test_spanish_identity(self, tok, text):
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text, f"Expected {text!r}, got {decoded!r}"

    @skip_no_model
    def test_numbers_identity(self, tok):
        for text in ["42", "3.14", "100", "0", "999"]:
            ids = tok.encode(text)
            decoded = tok.decode(ids)
            assert decoded == text

    @skip_no_model
    def test_mixed_language_identity(self, tok):
        text = "El modelo usa attention con F1=0.85"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    @skip_no_model
    def test_encode_returns_list_of_ints(self, tok):
        ids = tok.encode("test sentence")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        assert all(0 <= i < tok.vocab_size for i in ids)

    @skip_no_model
    def test_decode_returns_string(self, tok):
        ids = tok.encode("test")
        result = tok.decode(ids)
        assert isinstance(result, str)

    @skip_no_model
    def test_empty_string_encode(self, tok):
        ids = tok.encode("")
        assert ids == []

    @skip_no_model
    def test_empty_list_decode(self, tok):
        text = tok.decode([])
        assert text == ""

    @skip_no_model
    def test_decode_filters_special_tokens(self, tok):
        # pad, bos, eos deben ser ignorados al decodificar
        ids = [tok.bos_id] + tok.encode("hello") + [tok.eos_id]
        decoded = tok.decode(ids)
        assert decoded == "hello"

    @skip_no_model
    def test_decode_filters_pad(self, tok):
        ids = tok.encode("hi") + [tok.pad_id, tok.pad_id]
        decoded = tok.decode(ids)
        assert decoded == "hi"


# ─────────────────────────────────────────────────────────────────────────────
# Tests de add_bos / add_eos
# ─────────────────────────────────────────────────────────────────────────────

class TestBosEos:
    @skip_no_model
    def test_add_bos(self, tok):
        ids = tok.encode("hello", add_bos=True)
        assert ids[0] == tok.bos_id

    @skip_no_model
    def test_add_eos(self, tok):
        ids = tok.encode("hello", add_eos=True)
        assert ids[-1] == tok.eos_id

    @skip_no_model
    def test_add_bos_and_eos(self, tok):
        ids = tok.encode("hello world", add_bos=True, add_eos=True)
        assert ids[0] == tok.bos_id
        assert ids[-1] == tok.eos_id
        assert len(ids) >= 3  # bos + tokens + eos

    @skip_no_model
    def test_bos_eos_roundtrip(self, tok):
        text = "test roundtrip"
        ids = tok.encode(text, add_bos=True, add_eos=True)
        decoded = tok.decode(ids)
        assert decoded == text  # decode filtra bos/eos automáticamente


# ─────────────────────────────────────────────────────────────────────────────
# Tests de tokenize (piezas)
# ─────────────────────────────────────────────────────────────────────────────

class TestTokenize:
    @skip_no_model
    def test_tokenize_returns_list_of_strings(self, tok):
        pieces = tok.tokenize("hello world")
        assert isinstance(pieces, list)
        assert all(isinstance(p, str) for p in pieces)

    @skip_no_model
    def test_tokenize_nonempty(self, tok):
        pieces = tok.tokenize("hello")
        assert len(pieces) >= 1

    @skip_no_model
    def test_piece_to_id_roundtrip(self, tok):
        for i in [5, 100, 500, 1000, 5000, 10000, 31999]:
            piece = tok.id_to_piece(i)
            recovered_id = tok.piece_to_id(piece)
            assert recovered_id == i, f"id={i} piece={piece!r} recovered={recovered_id}"


# ─────────────────────────────────────────────────────────────────────────────
# Tests de batch
# ─────────────────────────────────────────────────────────────────────────────

class TestBatch:
    @skip_no_model
    def test_encode_batch(self, tok):
        texts = ["hello world", "Hola mundo", "test"]
        batch = tok.encode_batch(texts)
        assert len(batch) == len(texts)
        for ids, text in zip(batch, texts):
            assert ids == tok.encode(text)

    @skip_no_model
    def test_decode_batch(self, tok):
        texts = ["hello world", "Hola mundo", "test"]
        batch = tok.encode_batch(texts)
        decoded = tok.decode_batch(batch)
        for original, recovered in zip(texts, decoded):
            assert recovered == original

    @skip_no_model
    def test_batch_empty(self, tok):
        assert tok.encode_batch([]) == []
        assert tok.decode_batch([]) == []


# ─────────────────────────────────────────────────────────────────────────────
# Tests de coverage
# ─────────────────────────────────────────────────────────────────────────────

class TestCoverage:
    @skip_no_model
    def test_coverage_english(self, tok):
        text = "The neural network processes attention mechanisms for language modeling."
        cov = tok.coverage(text)
        assert cov >= 0.99, f"Coverage too low: {cov:.4f}"

    @skip_no_model
    def test_coverage_spanish(self, tok):
        text = "El sistema analiza los patrones morfologicos de las oraciones en el corpus."
        cov = tok.coverage(text)
        assert cov >= 0.99, f"Coverage too low: {cov:.4f}"

    @skip_no_model
    def test_coverage_empty(self, tok):
        assert tok.coverage("") == 1.0

    @skip_no_model
    @pytest.mark.skipif(not corpus_available(), reason="corpus not available")
    def test_coverage_on_corpus_sample(self, tok):
        """Verifica cobertura ≥ 99% en muestra real del corpus."""
        total = 0
        unk_count = 0
        sample_file = CORPUS_DIR / "mose_cora.jsonl"
        with open(sample_file, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 100:
                    break
                record = json.loads(line)
                for field in ("input", "expected_output", "reasoning"):
                    text = record.get(field, "")
                    if not text:
                        continue
                    ids = tok.encode(text)
                    total += len(ids)
                    unk_count += sum(1 for x in ids if x == tok.unk_id)

        coverage = 1.0 - unk_count / total if total > 0 else 1.0
        assert coverage >= 0.99, (
            f"Corpus coverage {coverage:.4f} < 0.99 "
            f"({unk_count} UNK out of {total} tokens)"
        )

    @skip_no_model
    @pytest.mark.skipif(not corpus_available(), reason="corpus not available")
    def test_coverage_all_motors(self, tok):
        """Cobertura ≥ 99% en los 6 motores del corpus."""
        motors = [
            "mose_cora.jsonl",
            "mose_forge_c.jsonl",
            "mose_axiom.jsonl",
            "mose_muse.jsonl",
            "mose_empathy.jsonl",
            "mose_orchestrator.jsonl",
        ]
        for fname in motors:
            path = CORPUS_DIR / fname
            if not path.exists():
                continue
            total = 0
            unk_count = 0
            with open(path, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 50:
                        break
                    record = json.loads(line)
                    text = record.get("reasoning", "")
                    if text:
                        ids = tok.encode(text)
                        total += len(ids)
                        unk_count += sum(1 for x in ids if x == tok.unk_id)
            if total > 0:
                cov = 1.0 - unk_count / total
                assert cov >= 0.99, (
                    f"{fname}: coverage={cov:.4f} < 0.99 "
                    f"({unk_count}/{total} UNK)"
                )


# ─────────────────────────────────────────────────────────────────────────────
# Tests de integración con StreamEncoder / StreamDecoder
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigIntegration:
    def test_stream_encoder_vocab_configurable(self):
        """StreamEncoderConfig acepta vocab_size desde config."""
        from encoder.mamba_layer import StreamEncoderConfig
        cfg = StreamEncoderConfig(vocab_size=32_000, hidden_dim=64, n_layers=2)
        assert cfg.vocab_size == 32_000

    def test_stream_decoder_vocab_configurable(self):
        """StreamDecoderConfig acepta vocab_size desde config."""
        from decoder.config import StreamDecoderConfig
        cfg = StreamDecoderConfig(vocab_size=32_000, hidden_dim=64, n_layers=2)
        assert cfg.vocab_size == 32_000

    def test_mose_scale_config_medium_vocab(self):
        """MoSEScaleConfig.medium() tiene vocab_size=32_000."""
        from router.config_3_5b import MoSEScaleConfig
        cfg = MoSEScaleConfig.medium()
        assert cfg.vocab_size == 32_000

    def test_mose_scale_config_production_vocab(self):
        """MoSEScaleConfig.production() tiene vocab_size=32_000."""
        from router.config_3_5b import MoSEScaleConfig
        cfg = MoSEScaleConfig.production()
        assert cfg.vocab_size == 32_000

    def test_stream_encoder_embedding_uses_vocab_size(self):
        """StreamEncoder usa config.vocab_size para el embedding."""
        import torch
        from encoder import StreamEncoder
        from encoder.mamba_layer import StreamEncoderConfig
        cfg = StreamEncoderConfig(vocab_size=100, hidden_dim=32, n_layers=1)
        enc = StreamEncoder(cfg)
        # Encontrar el embedding
        emb = None
        for mod in enc.modules():
            if isinstance(mod, torch.nn.Embedding):
                emb = mod
                break
        assert emb is not None
        assert emb.num_embeddings == 100

    def test_stream_decoder_lm_head_uses_vocab_size(self):
        """StreamDecoder usa config.vocab_size para la cabeza LM."""
        import torch
        from decoder.model import StreamDecoder
        from decoder.config import StreamDecoderConfig
        cfg = StreamDecoderConfig(vocab_size=100, hidden_dim=32, n_layers=1)
        dec = StreamDecoder(cfg)
        # Encontrar la capa Linear de la cabeza LM
        linear = None
        for name, mod in dec.named_modules():
            if isinstance(mod, torch.nn.Linear) and mod.out_features == 100:
                linear = mod
                break
        assert linear is not None

    @skip_no_model
    def test_tokenizer_vocab_matches_config_default(self, tok):
        """El vocab_size del tokenizador coincide con el default de las configs."""
        from encoder.mamba_layer import StreamEncoderConfig
        from decoder.config import StreamDecoderConfig
        enc_cfg = StreamEncoderConfig()
        dec_cfg = StreamDecoderConfig()
        assert tok.vocab_size == enc_cfg.vocab_size
        assert tok.vocab_size == dec_cfg.vocab_size


# ─────────────────────────────────────────────────────────────────────────────
# Tests de BPETrainer (entrenamiento en corpus pequeño)
# ─────────────────────────────────────────────────────────────────────────────

class TestBPETrainer:
    def test_trainer_instantiation(self):
        from tokenizer.bpe_tokenizer import BPETrainer
        trainer = BPETrainer(corpus_dir="/nonexistent", output_model="/tmp/test.model")
        assert trainer.vocab_size == 32_000

    def test_trainer_custom_vocab_size(self):
        from tokenizer.bpe_tokenizer import BPETrainer
        trainer = BPETrainer(corpus_dir="/nonexistent", vocab_size=1000)
        assert trainer.vocab_size == 1000

    def test_trainer_trains_small_corpus(self, tmp_path):
        """Entrena un tokenizador pequeño desde un corpus mínimo."""
        import sentencepiece as spm
        from tokenizer.bpe_tokenizer import BPETrainer, AIONTokenizer

        # Crear corpus pequeño con suficiente diversidad
        corpus_dir = tmp_path / "data"
        corpus_dir.mkdir()
        jsonl_file = corpus_dir / "mose_cora.jsonl"

        sentences = (
            "hello world this is a test sentence for tokenizer training\n"
            "hola mundo esta es una oracion de prueba para el tokenizador\n"
            "the quick brown fox jumps over the lazy dog\n"
            "el rapido zorro marron salta sobre el perro perezoso\n"
        )
        # Generar suficiente diversidad para vocab_size=200
        with open(jsonl_file, "w", encoding="utf-8") as f:
            for i in range(50):
                record = {
                    "input": sentences * 2,
                    "expected_output": f"output sentence number {i} with some words",
                    "reasoning": f"because reason number {i} applies here logically",
                }
                f.write(json.dumps(record) + "\n")

        model_path = tmp_path / "test.model"
        trainer = BPETrainer(
            corpus_dir=str(corpus_dir),
            output_model=str(model_path),
            vocab_size=200,
        )
        trainer.train(verbose=False)

        assert model_path.exists()
        tok = AIONTokenizer(model_path)
        assert tok.vocab_size == 200
        assert tok.pad_id == 0
        assert tok.bos_id == 1
        assert tok.eos_id == 2

        # Encode/decode básico
        text = "hello world"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_trainer_missing_files_ok(self, tmp_path):
        """BPETrainer omite archivos que no existen sin crash."""
        from tokenizer.bpe_tokenizer import BPETrainer, AIONTokenizer
        # Solo crear un archivo del corpus (faltan los otros 5)
        corpus_dir = tmp_path / "partial"
        corpus_dir.mkdir()
        with open(corpus_dir / "mose_cora.jsonl", "w", encoding="utf-8") as f:
            for i in range(20):
                f.write(json.dumps({
                    "input": f"test sentence {i} with words and more text",
                    "expected_output": f"output {i}",
                    "reasoning": f"reason {i} analysis complete done",
                }) + "\n")

        model_path = tmp_path / "partial.model"
        trainer = BPETrainer(
            corpus_dir=str(corpus_dir),
            output_model=str(model_path),
            vocab_size=100,
        )
        trainer.train(verbose=False)
        assert model_path.exists()


# ─────────────────────────────────────────────────────────────────────────────
# Tests adicionales de robustez
# ─────────────────────────────────────────────────────────────────────────────

class TestRobustness:
    @skip_no_model
    def test_long_text(self, tok):
        text = "This is a long text. " * 100
        text = text.strip()
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    @skip_no_model
    def test_single_char(self, tok):
        for c in "aeiouAEIOU":
            ids = tok.encode(c)
            assert len(ids) >= 1

    @skip_no_model
    def test_all_ids_in_range(self, tok):
        texts = [
            "hello world testing tokenizer",
            "El sistema de analisis funciona correctamente con todos los datos",
            "123 456 789 0.5 3.14 100%",
        ]
        for text in texts:
            ids = tok.encode(text)
            for i in ids:
                assert 0 <= i < tok.vocab_size, f"ID {i} out of range"

    @skip_no_model
    def test_model_path_property(self, tok):
        assert tok.model_path == MODEL_PATH

    @skip_no_model
    def test_encode_no_extra_bos_eos_by_default(self, tok):
        """Por defecto encode no añade bos/eos."""
        ids = tok.encode("hello")
        assert tok.bos_id not in ids
        assert tok.eos_id not in ids
