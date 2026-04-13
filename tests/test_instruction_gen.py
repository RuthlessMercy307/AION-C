"""Tests for synth/instruction_gen.py"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from synth.instruction_gen import InstructionGenerator


@pytest.fixture
def gen():
    return InstructionGenerator(seed=42)


class TestIdentity:
    def test_generates_correct_count(self, gen):
        examples = gen.gen_identity(100)
        assert len(examples) == 100

    def test_has_required_fields(self, gen):
        examples = gen.gen_identity(10)
        for ex in examples:
            assert "instruction" in ex
            assert "response" in ex
            assert ex["category"] == "identity"

    def test_mentions_aion_c(self, gen):
        examples = gen.gen_identity(50)
        aion_count = sum(1 for ex in examples if "AION-C" in ex["response"] or "AION" in ex["response"])
        assert aion_count > 0


class TestCasual:
    def test_generates(self, gen):
        examples = gen.gen_casual(100)
        assert len(examples) == 100

    def test_no_emojis(self, gen):
        examples = gen.gen_casual(50)
        emoji_chars = set("😀😁😂🤣😃😄😅😆😉😊😋😎")
        for ex in examples:
            assert not any(c in ex["response"] for c in emoji_chars)


class TestCode:
    def test_generates(self, gen):
        examples = gen.gen_code(100)
        assert len(examples) == 100

    def test_has_code_blocks(self, gen):
        examples = gen.gen_code(20)
        has_code = sum(1 for ex in examples if "```" in ex["response"])
        assert has_code > 0, "No code blocks found in code examples"


class TestMath:
    def test_generates(self, gen):
        examples = gen.gen_math(100)
        assert len(examples) == 100


class TestCreativity:
    def test_generates(self, gen):
        examples = gen.gen_creativity(50)
        assert len(examples) == 50


class TestSocial:
    def test_generates(self, gen):
        examples = gen.gen_social(50)
        assert len(examples) == 50


class TestAutonomy:
    def test_generates(self, gen):
        examples = gen.gen_autonomy(50)
        assert len(examples) == 50
        for ex in examples:
            assert ex["category"] == "autonomy"


class TestSelfVerify:
    def test_generates(self, gen):
        examples = gen.gen_self_verify(50)
        assert len(examples) == 50


class TestThinkingAloud:
    def test_generates(self, gen):
        examples = gen.gen_thinking_aloud(50)
        assert len(examples) == 50


class TestProactive:
    def test_generates(self, gen):
        examples = gen.gen_proactive(50)
        assert len(examples) == 50


class TestFormat:
    def test_generates(self, gen):
        examples = gen.gen_format(50)
        assert len(examples) == 50

    def test_has_markdown(self, gen):
        examples = gen.gen_format(20)
        has_md = sum(1 for ex in examples if "##" in ex["response"] or "```" in ex["response"])
        assert has_md > 0


class TestSafety:
    def test_generates(self, gen):
        examples = gen.gen_safety(50)
        assert len(examples) == 50

    def test_refuses(self, gen):
        examples = gen.gen_safety(20)
        for ex in examples:
            resp = ex["response"].lower()
            assert any(w in resp for w in ["no puedo", "no voy", "i can't", "i won't", "contra mis", "goes against", "that goes"]), \
                f"Safety response doesn't refuse: {resp[:100]}"


class TestMetacognition:
    def test_generates(self, gen):
        examples = gen.gen_metacognition(50)
        assert len(examples) == 50


class TestMemUsage:
    def test_generates(self, gen):
        examples = gen.gen_mem_usage(50)
        assert len(examples) == 50

    def test_has_mem_tags(self, gen):
        examples = gen.gen_mem_usage(50)
        has_search = sum(1 for ex in examples if "BUSCAR_MEM" in ex["response"])
        has_store = sum(1 for ex in examples if "GUARDAR_MEM" in ex["response"])
        assert has_search + has_store > 0, "No MEM tags found"


class TestMultiTurn:
    def test_generates(self, gen):
        examples = gen.gen_multi_turn(50)
        assert len(examples) == 50

    def test_has_conversation(self, gen):
        examples = gen.gen_multi_turn(20)
        for ex in examples:
            assert "conversation" in ex
            assert len(ex["conversation"]) >= 3, "Conversation too short"


class TestSystemPrompt:
    def test_generates(self, gen):
        examples = gen.gen_system_prompt(50)
        assert len(examples) == 50

    def test_has_system_prompt(self, gen):
        examples = gen.gen_system_prompt(20)
        for ex in examples:
            assert "system_prompt" in ex
            assert len(ex["system_prompt"]) > 0


class TestGenerateAll:
    def test_total_count(self, gen):
        examples = gen.generate_all()
        assert len(examples) >= 25000, f"Expected >= 25000, got {len(examples)}"

    def test_categories_present(self, gen):
        examples = gen.generate_all()
        categories = set(ex["category"] for ex in examples)
        expected = {
            "identity", "casual", "reasoning", "code", "math", "creativity",
            "social", "autonomy", "self_verify", "thinking_aloud", "proactive",
            "format", "safety", "metacognition", "mem_usage", "multi_turn",
            "system_prompt",
        }
        assert categories == expected, f"Missing categories: {expected - categories}"
