"""Financial situation memory using BM25 for lexical similarity matching.

Uses BM25 (Best Matching 25) algorithm for retrieval - no API calls,
no token limits, works offline with any LLM provider.

Supports persistent storage via save/load to JSON files.
"""

import json
import os
import logging
from rank_bm25 import BM25Okapi
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)
import re


class FinancialSituationMemory:
    """Memory system for storing and retrieving financial situations using BM25.

    Supports persistent storage — call save() to persist to disk and load() to restore.
    """

    def __init__(self, name: str, config: dict = None):
        """Initialize the memory system.

        Args:
            name: Name identifier for this memory instance
            config: Configuration dict. If it contains 'memory_dir', auto-loads from disk.
        """
        self.name = name
        self.config = config or {}
        self.documents: List[str] = []
        self.recommendations: List[str] = []
        self.bm25 = None
        self._memory_path: Optional[str] = None

        # Auto-load from disk if memory_dir is configured
        memory_dir = self.config.get("memory_dir") or self.config.get("results_dir")
        if memory_dir:
            self._memory_path = os.path.join(
                os.path.abspath(memory_dir), "memories", f"{name}.json"
            )
            self.load()

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25 indexing.

        Simple whitespace + punctuation tokenization with lowercasing.
        """
        # Lowercase and split on non-alphanumeric characters
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def _rebuild_index(self):
        """Rebuild the BM25 index after adding documents."""
        if self.documents:
            tokenized_docs = [self._tokenize(doc) for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_docs)
        else:
            self.bm25 = None

    def add_situations(self, situations_and_advice: List[Tuple[str, str]]):
        """Add financial situations and their corresponding advice.

        Args:
            situations_and_advice: List of tuples (situation, recommendation)
        """
        for situation, recommendation in situations_and_advice:
            self.documents.append(situation)
            self.recommendations.append(recommendation)

        # Rebuild BM25 index with new documents
        self._rebuild_index()

        # Auto-save if persistence is configured
        if self._memory_path:
            self.save()

    def get_memories(self, current_situation: str, n_matches: int = 1) -> List[dict]:
        """Find matching recommendations using BM25 similarity.

        Args:
            current_situation: The current financial situation to match against
            n_matches: Number of top matches to return

        Returns:
            List of dicts with matched_situation, recommendation, and similarity_score
        """
        if not self.documents or self.bm25 is None:
            return []

        # Tokenize query
        query_tokens = self._tokenize(current_situation)

        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(query_tokens)

        # Get top-n indices sorted by score (descending)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_matches]

        # Build results — use raw BM25 scores with absolute threshold
        # to avoid returning irrelevant matches from small corpora
        MIN_BM25_SCORE = 1.0  # Minimum raw BM25 score to consider relevant
        results = []

        for idx in top_indices:
            raw_score = scores[idx]
            if raw_score < MIN_BM25_SCORE:
                continue
            results.append({
                "matched_situation": self.documents[idx],
                "recommendation": self.recommendations[idx],
                "similarity_score": raw_score,
            })

        return results

    def save(self, path: Optional[str] = None):
        """Persist memories to a JSON file.

        Args:
            path: File path to save to. If None, uses the auto-configured path.
        """
        save_path = path or self._memory_path
        if not save_path:
            return

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        data = {
            "name": self.name,
            "documents": self.documents,
            "recommendations": self.recommendations,
        }
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.debug("Saved %d memories to %s", len(self.documents), save_path)
        except OSError as e:
            logger.warning("Failed to save memories to %s: %s", save_path, e)

    def load(self, path: Optional[str] = None):
        """Load memories from a JSON file.

        Args:
            path: File path to load from. If None, uses the auto-configured path.
        """
        load_path = path or self._memory_path
        if not load_path or not os.path.exists(load_path):
            return

        try:
            with open(load_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.documents = data.get("documents", [])
            self.recommendations = data.get("recommendations", [])
            self._rebuild_index()
            logger.debug("Loaded %d memories from %s", len(self.documents), load_path)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load memories from %s: %s", load_path, e)

    def clear(self):
        """Clear all stored memories."""
        self.documents = []
        self.recommendations = []
        self.bm25 = None


if __name__ == "__main__":
    # Example usage
    matcher = FinancialSituationMemory("test_memory")

    # Example data
    example_data = [
        (
            "High inflation rate with rising interest rates and declining consumer spending",
            "Consider defensive sectors like consumer staples and utilities. Review fixed-income portfolio duration.",
        ),
        (
            "Tech sector showing high volatility with increasing institutional selling pressure",
            "Reduce exposure to high-growth tech stocks. Look for value opportunities in established tech companies with strong cash flows.",
        ),
        (
            "Strong dollar affecting emerging markets with increasing forex volatility",
            "Hedge currency exposure in international positions. Consider reducing allocation to emerging market debt.",
        ),
        (
            "Market showing signs of sector rotation with rising yields",
            "Rebalance portfolio to maintain target allocations. Consider increasing exposure to sectors benefiting from higher rates.",
        ),
    ]

    # Add the example situations and recommendations
    matcher.add_situations(example_data)

    # Example query
    current_situation = """
    Market showing increased volatility in tech sector, with institutional investors
    reducing positions and rising interest rates affecting growth stock valuations
    """

    try:
        recommendations = matcher.get_memories(current_situation, n_matches=2)

        for i, rec in enumerate(recommendations, 1):
            print(f"\nMatch {i}:")
            print(f"Similarity Score: {rec['similarity_score']:.2f}")
            print(f"Matched Situation: {rec['matched_situation']}")
            print(f"Recommendation: {rec['recommendation']}")

    except Exception as e:
        print(f"Error during recommendation: {str(e)}")
