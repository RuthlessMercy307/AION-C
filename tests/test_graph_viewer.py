"""Tests for visualization/graph_viewer.py"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from core.graph import CausalGraph, CausalNode, CausalEdge, NodeType, CausalRelation
from visualization.graph_viewer import GraphViewer, ascii_graph, html_graph, ascii_scratch_pad


@pytest.fixture
def sample_graph():
    g = CausalGraph()
    g.add_node(CausalNode(node_id="a", label="Rain", node_type=NodeType.ENTITY, confidence=0.9))
    g.add_node(CausalNode(node_id="b", label="Flood", node_type=NodeType.EVENT, confidence=0.7))
    g.add_edge(CausalEdge(source_id="a", target_id="b", relation=CausalRelation.CAUSES, strength=0.8))
    return g


class TestAsciiGraph:
    def test_renders_nodes(self, sample_graph):
        text = ascii_graph(sample_graph)
        assert "Rain" in text
        assert "Flood" in text

    def test_renders_edges(self, sample_graph):
        text = ascii_graph(sample_graph)
        assert "-->" in text
        assert "causes" in text.lower() or "CAUSES" in text

    def test_shows_counts(self, sample_graph):
        text = ascii_graph(sample_graph)
        assert "Nodes: 2" in text
        assert "Edges: 1" in text

    def test_empty_graph(self):
        g = CausalGraph()
        text = ascii_graph(g)
        assert "Nodes: 0" in text

    def test_custom_title(self, sample_graph):
        text = ascii_graph(sample_graph, title="My Graph")
        assert "My Graph" in text


class TestHtmlGraph:
    def test_generates_valid_html(self, sample_graph):
        html = html_graph(sample_graph)
        assert "<!DOCTYPE html>" in html or "<html>" in html
        assert "vis-network" in html
        assert "Rain" in html
        assert "Flood" in html

    def test_saves_to_file(self, sample_graph, tmp_path):
        out = str(tmp_path / "test_graph.html")
        html = html_graph(sample_graph, output_path=out)
        assert os.path.exists(out)
        with open(out) as f:
            content = f.read()
        assert "Rain" in content

    def test_with_scratch_pad(self, sample_graph):
        states = [torch.randn(4, 8), torch.randn(4, 8)]
        html = html_graph(sample_graph, pad_states=states)
        assert "Scratch Pad" in html or "scratch" in html.lower()


class TestAsciiScratchPad:
    def test_renders_slots(self):
        states = [torch.randn(4, 8)]
        text = ascii_scratch_pad(states)
        assert "slot_0" in text or "Iteration 0" in text

    def test_multiple_iterations(self):
        states = [torch.randn(4, 8) for _ in range(3)]
        text = ascii_scratch_pad(states)
        assert "Iteration 0" in text
        assert "Iteration 2" in text

    def test_custom_labels(self):
        states = [torch.randn(2, 8)]
        text = ascii_scratch_pad(states, slot_labels=["mem_A", "mem_B"])
        assert "mem_A" in text


class TestGraphViewer:
    def test_to_ascii(self, sample_graph):
        viewer = GraphViewer(sample_graph, title="Test")
        text = viewer.to_ascii()
        assert "Test" in text
        assert "Rain" in text

    def test_to_html(self, sample_graph):
        viewer = GraphViewer(sample_graph)
        html = viewer.to_html()
        assert "<html>" in html.lower() or "<!doctype" in html.lower()

    def test_with_pad_states(self, sample_graph):
        states = [torch.randn(4, 8)]
        viewer = GraphViewer(sample_graph, pad_states=states)
        text = viewer.to_ascii()
        assert "SCRATCH PAD" in text
