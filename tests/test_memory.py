"""Tests for FinancialSituationMemory — BM25 retrieval, dedup, eviction, persistence."""

import json
import os
import pytest

from tradingagents.agents.utils.memory import FinancialSituationMemory


@pytest.fixture
def mem():
    return FinancialSituationMemory("test")


@pytest.fixture
def mem_with_data(mem):
    mem.add_situations([
        ("High inflation with rising interest rates", "Consider defensive sectors"),
        ("Tech sector showing high volatility", "Reduce exposure to growth stocks"),
        ("Strong dollar affecting emerging markets", "Hedge currency exposure"),
    ])
    return mem


# ── Basic add / retrieve ────────────────────────────────────────────

class TestBasicOperations:
    def test_add_and_count(self, mem):
        mem.add_situations([("sit1", "rec1"), ("sit2", "rec2")])
        assert len(mem.documents) == 2
        assert len(mem.recommendations) == 2

    def test_empty_memory_returns_no_matches(self, mem):
        assert mem.get_memories("anything") == []

    def test_retrieve_after_add(self, mem_with_data):
        results = mem_with_data.get_memories("inflation and interest rates", n_matches=1)
        assert len(results) >= 1
        assert "inflation" in results[0]["matched_situation"].lower()

    def test_retrieve_multiple_matches(self, mem_with_data):
        results = mem_with_data.get_memories(
            "High inflation rising interest rates tech sector volatility emerging markets dollar",
            n_matches=3,
        )
        assert len(results) >= 1
        for r in results:
            assert "matched_situation" in r
            assert "recommendation" in r
            assert "similarity_score" in r

    def test_bm25_relevance_ordering(self, mem_with_data):
        results = mem_with_data.get_memories("tech sector volatility", n_matches=3)
        if len(results) >= 2:
            assert results[0]["similarity_score"] >= results[1]["similarity_score"]


# ── Deduplication ───────────────────────────────────────────────────

class TestDeduplication:
    def test_exact_duplicate_rejected(self, mem):
        mem.add_situations([("A", "rec_a"), ("A", "rec_a")])
        assert len(mem.documents) == 1

    def test_same_situation_different_rec_accepted(self, mem):
        mem.add_situations([("A", "rec_1"), ("A", "rec_2")])
        assert len(mem.documents) == 2

    def test_duplicate_across_batches(self, mem):
        mem.add_situations([("A", "rec_a")])
        mem.add_situations([("A", "rec_a")])
        assert len(mem.documents) == 1

    def test_seen_set_stays_in_sync(self, mem):
        mem.add_situations([("X", "Y"), ("Z", "W")])
        assert len(mem._seen) == 2
        assert ("X", "Y") in mem._seen


# ── Eviction (max_size) ────────────────────────────────────────────

class TestEviction:
    def test_evicts_oldest_when_over_capacity(self):
        mem = FinancialSituationMemory("test", {"max_memory_size": 3})
        mem.add_situations([("A", "1"), ("B", "2"), ("C", "3")])
        mem.add_situations([("D", "4")])
        assert len(mem.documents) == 3
        assert mem.documents[0] == "B"  # A evicted

    def test_seen_set_updated_after_eviction(self):
        mem = FinancialSituationMemory("test", {"max_memory_size": 2})
        mem.add_situations([("A", "1"), ("B", "2")])
        mem.add_situations([("C", "3")])
        assert ("A", "1") not in mem._seen
        assert ("B", "2") in mem._seen
        assert ("C", "3") in mem._seen

    def test_evicted_entry_can_be_readded(self):
        mem = FinancialSituationMemory("test", {"max_memory_size": 2})
        mem.add_situations([("A", "1"), ("B", "2")])
        mem.add_situations([("C", "3")])  # evicts A
        mem.add_situations([("A", "1")])  # re-add A (no longer duplicate)
        assert "A" in mem.documents

    def test_default_max_size(self, mem):
        assert mem.max_size == 500


# ── Persistence (save / load) ──────────────────────────────────────

class TestPersistence:
    def test_save_and_load(self, tmp_path):
        path = str(tmp_path / "mem.json")
        mem1 = FinancialSituationMemory("test")
        mem1.add_situations([("sit1", "rec1"), ("sit2", "rec2")])
        mem1.save(path)

        mem2 = FinancialSituationMemory("test")
        mem2.load(path)
        assert mem2.documents == mem1.documents
        assert mem2.recommendations == mem1.recommendations
        assert mem2._seen == mem1._seen

    def test_load_rebuilds_bm25_index(self, tmp_path):
        path = str(tmp_path / "mem.json")
        mem1 = FinancialSituationMemory("test")
        mem1.add_situations([("inflation rates rising", "buy bonds")])
        mem1.save(path)

        mem2 = FinancialSituationMemory("test")
        mem2.load(path)
        results = mem2.get_memories("inflation", n_matches=1)
        assert len(results) >= 1

    def test_auto_save_with_memory_dir(self, tmp_path):
        config = {"memory_dir": str(tmp_path)}
        mem = FinancialSituationMemory("auto_test", config)
        mem.add_situations([("test", "rec")])
        expected_path = tmp_path / "memories" / "auto_test.json"
        assert expected_path.exists()

    def test_load_nonexistent_file_is_noop(self, mem):
        mem.load("/nonexistent/path.json")
        assert mem.documents == []


# ── Clear ───────────────────────────────────────────────────────────

class TestClear:
    def test_clear_empties_all(self, mem_with_data):
        mem_with_data.clear()
        assert mem_with_data.documents == []
        assert mem_with_data.recommendations == []
        assert mem_with_data._seen == set()
        assert mem_with_data.bm25 is None
