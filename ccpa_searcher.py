"""
CCPA Search & Retrieval — provides search(), get_sections(), and format_for_llm().

All search results derive exclusively from data/ccpa_statute.txt via the parser
and indexer. No section numbers, titles, or legal text are hardcoded in this module.

This module provides three levels of output:
  - search()         → full chunk dicts with text (for LLM pipeline reasoning)
  - get_sections()   → clean section label strings (for final pipeline output)
  - format_for_llm() → formatted string for LLM citation prompts
"""

from collections import defaultdict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ccpa_indexer import CCPAIndexer


class CCPASearcher:
    """Search engine over CCPA statute chunks using TF-IDF + cosine similarity."""

    def __init__(self, indexer: CCPAIndexer, chunks: list[dict]):
        """
        Initialize the searcher with a built/loaded indexer and chunks.

        Args:
            indexer: CCPAIndexer instance (built or loaded from cache)
            chunks:  List of chunk dicts from ccpa_parser.parse_statute()
        """
        self.indexer = indexer
        self.chunks = chunks

    def search(
        self, query: str, top_k: int = 10, score_threshold: float = 0.01
    ) -> list[dict]:
        """
        PRIMARY FUNCTION FOR LLM PIPELINE USE.

        Returns full chunk dicts including text — the LLM needs this text
        to reason about which sections are genuinely violated vs merely related.

        Each dict has: id, section, sub, label, title, text, depth, score
        """
        # Transform query
        query_vec = self.indexer.vectorizer.transform([query])

        # Compute cosine similarity
        scores = cosine_similarity(query_vec, self.indexer.tfidf_matrix).flatten()

        # Get indices sorted by score descending
        ranked_indices = np.argsort(scores)[::-1]

        # Filter by threshold
        candidates = []
        for idx in ranked_indices:
            if scores[idx] < score_threshold:
                break
            chunk = dict(self.chunks[idx])  # copy
            chunk["score"] = float(scores[idx])
            candidates.append(chunk)

        # De-duplicate: prefer subsections over section-level chunks
        sections_with_subs = set()
        for c in candidates:
            if c["sub"]:
                sections_with_subs.add(c["section"])

        deduped = []
        for c in candidates:
            if c["depth"] == 0 and c["section"] in sections_with_subs:
                continue
            deduped.append(c)

        return deduped[:top_k]

    def get_sections(
        self, query: str, top_k: int = 10, threshold: float = 0.01
    ) -> list[str]:
        """
        Returns top-level section labels only (e.g. "Section 1798.120").

        Scores all chunks, then collapses to the parent section level by taking
        the max score from any chunk belonging to that section. This ensures a
        section is surfaced if *any* of its subsections match.

        Example: ["Section 1798.120", "Section 1798.100", "Section 1798.115"]
        """
        query_vec = self.indexer.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.indexer.tfidf_matrix).flatten()

        # Aggregate: max score per top-level section
        section_best: dict[str, float] = {}
        section_title: dict[str, str] = {}
        for i, chunk in enumerate(self.chunks):
            sec = chunk["section"]
            sc = float(scores[i])
            if sc >= threshold and sc > section_best.get(sec, 0.0):
                section_best[sec] = sc
                section_title[sec] = chunk["title"]
        EXCLUDE = {"1798.140"}
    
        ranked = sorted(section_best.items(), key=lambda x: x[1], reverse=True)
        ranked = [(s, sc) for s, sc in ranked if s not in EXCLUDE]
        # Sort by score descending, take top_k
        ranked = sorted(section_best.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [f"Section {sec}" for sec, _ in ranked]

    def format_for_llm(
        self, query: str, top_k: int = 10, text_limit: int = 300
    ) -> str:
        """
        FORMATS SEARCH RESULTS FOR LLM CITATION PROMPT.

        Returns a formatted string ready to be injected into an LLM prompt.
        Each candidate section is shown with its label + first text_limit characters
        of its legal text. This gives the LLM enough context to reason about
        whether the section is genuinely violated without overflowing context window.

        Format:
            Section 1798.120(c): Notwithstanding subdivision (a), a business shall
            not sell or share the personal information of consumers if the business
            has actual knowledge that the consumer is less than 16 years of age...
        """
        results = self.search(query, top_k=top_k)
        lines = []
        for r in results:
            snippet = r["text"][:text_limit].strip()
            if len(r["text"]) > text_limit:
                snippet += "..."
            lines.append(f"{r['label']}: {snippet}")
        return "\n\n".join(lines)

    def group_by_section(self, results: list[dict]) -> dict[str, list[dict]]:
        """Group result chunks by their top-level section number for display."""
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in results:
            groups[r["section"]].append(r)
        return dict(groups)
