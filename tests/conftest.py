from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
CORPUS = FIXTURES / "corpus"


def word_count(text: str) -> int:
    """Stand-in token counter: one token per whitespace-separated word.
    The real counter is the embedder's tokenizer; the chunkers only need a
    monotone count."""
    return len(text.split())


class HashEmbedder:
    """Deterministic fake embedder: a bag of hashed words, L2-normalised."""

    dim = 64
    max_seq_length = 256

    def encode(self, texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().split():
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
            norm = np.linalg.norm(out[i])
            if norm:
                out[i] /= norm
        return out


@pytest.fixture
def count():
    return word_count


@pytest.fixture
def embedder():
    return HashEmbedder()


@pytest.fixture
def fixture_corpus() -> Path:
    return CORPUS


@pytest.fixture
def read_fixture():
    def _read(rel: str) -> str:
        return (CORPUS / "demo" / rel).read_text(encoding="utf-8")

    return _read
