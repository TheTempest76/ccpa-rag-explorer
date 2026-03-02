"""
CCPA TF-IDF Indexer — builds and caches a TF-IDF index from parsed statute chunks.

All indexed content derives exclusively from data/ccpa_statute.txt via the parser.
No section numbers, titles, or legal text are hardcoded in this module.
The index is saved as a pickle file to index/ccpa_index.pkl for fast reloading.
"""

import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class CCPAIndexer:
    """Builds and manages a TF-IDF index over CCPA statute chunks."""

    def __init__(self, chunks: list[dict]):
        """
        Build TF-IDF index from parsed chunks.

        For each chunk, the document string is constructed as:
            "{label} {title} {title} {text}"
        Title is doubled to give section headings extra TF-IDF weight.

        Args:
            chunks: List of chunk dicts from ccpa_parser.parse_statute()
        """
        self.chunks = chunks

        # Build document strings — title doubled for extra weight
        self.documents = []
        for chunk in chunks:
            doc = f"{chunk['label']} {chunk['title']} {chunk['title']} {chunk['text']}"
            self.documents.append(doc)

        # Build TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            stop_words="english",
        )

        # Fit and transform
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)

    def save(self, path: Path) -> None:
        """Save the index (vectorizer, matrix, chunks) to a pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "vectorizer": self.vectorizer,
            "tfidf_matrix": self.tfidf_matrix,
            "chunks": self.chunks,
            "documents": self.documents,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> "CCPAIndexer":
        """
        Load a previously saved index from pickle.

        Args:
            path: Path to the pickle file (e.g. index/ccpa_index.pkl)

        Returns:
            Reconstructed CCPAIndexer instance.
        """
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls.__new__(cls)
        instance.vectorizer = data["vectorizer"]
        instance.tfidf_matrix = data["tfidf_matrix"]
        instance.chunks = data["chunks"]
        instance.documents = data["documents"]
        return instance


if __name__ == "__main__":
    from ccpa_parser import parse_statute

    base = Path(__file__).parent
    chunks = parse_statute(base / "data" / "ccpa_statute.txt")
    indexer = CCPAIndexer(chunks)
    indexer.save(base / "index" / "ccpa_index.pkl")
    print(f"Index built and saved: {len(chunks)} chunks, "
          f"vocabulary size: {len(indexer.vectorizer.vocabulary_)}")
