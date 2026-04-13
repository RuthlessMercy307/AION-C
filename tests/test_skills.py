"""
tests/test_skills.py — Tests para agent/skills.py (Parte 3 del MEGA-PROMPT)
============================================================================

Cubre:
  Skill / SkillsLoader.load_file / load_dir / add_skill
  Frontmatter parser (sin dependencia de PyYAML)
  attach_to_mem con domain="skill"
  search() con threshold y top_k
  format_for_injection con bloque [SKILL: ...]
  Verificación de los 11 skills iniciales del MEGA-PROMPT (3.2)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from agent.skills import (
    Skill, SkillsLoader, SKILL_DOMAIN, _parse_frontmatter,
)


REPO = Path(__file__).resolve().parent.parent
SKILLS_DIR = REPO / "skills"


# ─────────────────────────────────────────────────────────────────────────────
# FakeMem con soporte de filtro por dominio
# ─────────────────────────────────────────────────────────────────────────────


class FakeMem:
    def __init__(self):
        self.entries = {}  # key -> (value, domain)

    def store(self, key, value, domain="general", source="test"):
        self.entries[key] = (value, domain)

    def search(self, query, top_k=5, domain=None):
        out = []
        for k, (v, d) in self.entries.items():
            if domain and d != domain:
                continue
            score = 1.0 if query.lower() in v.lower() or query.lower() in k.lower() else 0.4
            out.append((k, v, score))
        out.sort(key=lambda x: x[2], reverse=True)
        return out[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# Frontmatter parser
# ─────────────────────────────────────────────────────────────────────────────


class TestFrontmatter:
    def test_no_frontmatter(self):
        meta, body = _parse_frontmatter("just text")
        assert meta == {}
        assert body == "just text"

    def test_simple_frontmatter(self):
        text = "---\nname: foo\ndomain: forge_c\n---\nbody here"
        meta, body = _parse_frontmatter(text)
        assert meta["name"] == "foo"
        assert meta["domain"] == "forge_c"
        assert body == "body here"

    def test_quoted_strings(self):
        text = '---\nname: "with space"\n---\nbody'
        meta, _ = _parse_frontmatter(text)
        assert meta["name"] == "with space"

    def test_list_value(self):
        text = "---\ntags: [a, b, c]\n---\nbody"
        meta, _ = _parse_frontmatter(text)
        assert meta["tags"] == ["a", "b", "c"]

    def test_int_value(self):
        text = "---\npriority: 3\n---\nbody"
        meta, _ = _parse_frontmatter(text)
        assert meta["priority"] == 3
        assert isinstance(meta["priority"], int)

    def test_bool_value(self):
        text = "---\nactive: true\n---\nbody"
        meta, _ = _parse_frontmatter(text)
        assert meta["active"] is True

    def test_malformed_returns_unparsed(self):
        text = "---\nno close marker\nbody"
        meta, body = _parse_frontmatter(text)
        assert meta == {}
        assert body == text

    def test_empty_frontmatter(self):
        meta, body = _parse_frontmatter("---\n---\nbody")
        assert meta == {}
        assert body == "body"


# ─────────────────────────────────────────────────────────────────────────────
# SkillsLoader basic operations
# ─────────────────────────────────────────────────────────────────────────────


class TestLoader:
    def test_add_and_retrieve_skill(self):
        loader = SkillsLoader()
        s = loader.add_skill("test", "content here")
        assert len(loader) == 1
        assert "test" in loader.skills
        assert s.content == "content here"

    def test_load_file(self, tmp_path):
        f = tmp_path / "demo.md"
        f.write_text("---\nname: demo\ntags: [x, y]\n---\nthe body", encoding="utf-8")
        loader = SkillsLoader()
        skill = loader.load_file(f)
        assert skill.key == "demo"
        assert skill.content == "the body"
        assert skill.metadata["tags"] == ["x", "y"]
        assert len(loader) == 1

    def test_load_file_no_frontmatter(self, tmp_path):
        f = tmp_path / "raw.md"
        f.write_text("just markdown content\nline two", encoding="utf-8")
        loader = SkillsLoader()
        skill = loader.load_file(f)
        assert skill.key == "raw"
        assert "just markdown content" in skill.content

    def test_load_dir(self, tmp_path):
        (tmp_path / "a.md").write_text("a content", encoding="utf-8")
        (tmp_path / "b.md").write_text("b content", encoding="utf-8")
        (tmp_path / "ignore.txt").write_text("not a skill", encoding="utf-8")
        loader = SkillsLoader()
        loaded = loader.load_dir(tmp_path)
        assert len(loaded) == 2
        assert {s.key for s in loaded} == {"a", "b"}

    def test_load_dir_nonexistent_returns_empty(self, tmp_path):
        loader = SkillsLoader()
        assert loader.load_dir(tmp_path / "missing") == []

    def test_skill_title_from_key(self):
        s = Skill(key="empathetic_response", content="x")
        assert s.title == "Empathetic Response"


# ─────────────────────────────────────────────────────────────────────────────
# Persistencia en MEM
# ─────────────────────────────────────────────────────────────────────────────


class TestAttachToMem:
    def test_attach_writes_all_skills(self):
        loader = SkillsLoader()
        loader.add_skill("k1", "c1")
        loader.add_skill("k2", "c2")
        mem = FakeMem()
        n = loader.attach_to_mem(mem)
        assert n == 2
        assert mem.entries["k1"] == ("c1", SKILL_DOMAIN)
        assert mem.entries["k2"] == ("c2", SKILL_DOMAIN)

    def test_attach_to_none_is_safe(self):
        loader = SkillsLoader()
        loader.add_skill("k", "v")
        assert loader.attach_to_mem(None) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Búsqueda con threshold
# ─────────────────────────────────────────────────────────────────────────────


class TestSearch:
    def setup_method(self):
        self.loader = SkillsLoader()
        self.loader.add_skill("python_best_practices", "use type hints and f-strings")
        self.loader.add_skill("javascript_patterns", "use const and async await")
        self.loader.add_skill("identity", "I am AION-C")
        self.mem = FakeMem()
        self.loader.attach_to_mem(self.mem)

    def test_finds_relevant_skill(self):
        results = self.loader.search("type hints", self.mem)
        assert len(results) >= 1
        assert results[0][0] == "python_best_practices"

    def test_threshold_filters_low_scores(self):
        results = self.loader.search("type hints", self.mem, threshold=0.9)
        # Solo el match exacto debería pasar el threshold
        assert all(r[2] >= 0.9 for r in results)

    def test_top_k_limits(self):
        results = self.loader.search("AION-C", self.mem, top_k=1)
        assert len(results) <= 1

    def test_empty_query_returns_empty(self):
        assert self.loader.search("", self.mem) == []

    def test_no_mem_returns_empty(self):
        assert self.loader.search("anything", None) == []

    def test_only_searches_skill_domain(self):
        # Inserta una entrada en MEM con dominio distinto
        self.mem.store("user_pref", "type hints in Python", domain="user_model")
        results = self.loader.search("type hints", self.mem, top_k=10)
        keys = {r[0] for r in results}
        assert "user_pref" not in keys
        assert "python_best_practices" in keys


# ─────────────────────────────────────────────────────────────────────────────
# Format for injection
# ─────────────────────────────────────────────────────────────────────────────


class TestFormatForInjection:
    def test_formats_tuples(self):
        loader = SkillsLoader()
        out = loader.format_for_injection([("k1", "content one", 0.9), ("k2", "content two", 0.7)])
        assert "[SKILL: content one]" in out
        assert "[SKILL: content two]" in out

    def test_formats_skill_objects(self):
        loader = SkillsLoader()
        skills = [Skill(key="k", content="alpha")]
        out = loader.format_for_injection(skills)
        assert out == "[SKILL: alpha]"

    def test_empty_returns_empty_string(self):
        loader = SkillsLoader()
        assert loader.format_for_injection([]) == ""


# ─────────────────────────────────────────────────────────────────────────────
# Los 11 skills iniciales del MEGA-PROMPT 3.2
# ─────────────────────────────────────────────────────────────────────────────


REQUIRED_SKILLS = [
    "python_best_practices",
    "javascript_patterns",
    "causal_reasoning",
    "math_step_by_step",
    "creative_writing",
    "empathetic_response",
    "identity",
    "code_debugging",
    "spanish_responses",
    "sqlite_patterns",
    "web_development",
]


class TestInitialSkills:
    def test_all_11_files_exist(self):
        for name in REQUIRED_SKILLS:
            path = SKILLS_DIR / f"{name}.md"
            assert path.exists(), f"missing {path}"

    def test_load_all_initial_skills(self):
        loader = SkillsLoader()
        loaded = loader.load_dir(SKILLS_DIR)
        loaded_keys = {s.key for s in loaded}
        for name in REQUIRED_SKILLS:
            assert name in loaded_keys, f"didn't load {name}"

    def test_all_have_frontmatter_with_domain(self):
        loader = SkillsLoader()
        loader.load_dir(SKILLS_DIR)
        for name in REQUIRED_SKILLS:
            skill = loader.skills[name]
            assert "domain" in skill.metadata, f"{name} missing domain"
            assert skill.content.strip(), f"{name} has empty content"

    def test_attach_initial_skills_to_mem(self):
        loader = SkillsLoader()
        loader.load_dir(SKILLS_DIR)
        mem = FakeMem()
        n = loader.attach_to_mem(mem)
        assert n == len(REQUIRED_SKILLS)
        for name in REQUIRED_SKILLS:
            assert name in mem.entries
            assert mem.entries[name][1] == SKILL_DOMAIN
