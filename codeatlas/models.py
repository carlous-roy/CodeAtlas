"""Pinned models and their loaders.

Every model is loaded at a fixed Hugging Face revision so that a rerun sees the
same weights and the same tokenizer as the committed results.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder, SentenceTransformer


@dataclass(frozen=True)
class ModelPin:
    """A Hugging Face model name, the revision it is loaded at, and its input window in word pieces."""

    name: str
    revision: str
    max_tokens: int


EMBEDDER = ModelPin(
    name="sentence-transformers/all-MiniLM-L6-v2",
    revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    # The sentence-transformers configuration of this model truncates input at
    # 256 word pieces, including the [CLS] and [SEP] tokens.
    max_tokens=256,
)

RERANKER = ModelPin(
    name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    revision="233902d25c440f23af6f7d6e94d2946bac0bee0a",
    max_tokens=512,
)

# Tokens the encoder adds around the payload ([CLS] ... [SEP]).
SPECIAL_TOKENS = 2

TokenCounter = Callable[[str], int]


def _quiet() -> None:
    """Silence download progress bars and the tokenizer's length warning; the
    chunker measures lines longer than the window on purpose."""
    import os

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()


def load_embedder() -> SentenceTransformer:
    """The pinned sentence embedder on CPU; fails if its window is not the one the chunkers assume."""
    _quiet()
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDER.name, revision=EMBEDDER.revision, device="cpu")
    if model.max_seq_length != EMBEDDER.max_tokens:
        raise RuntimeError(
            f"{EMBEDDER.name} reports max_seq_length={model.max_seq_length}, "
            f"expected {EMBEDDER.max_tokens}; the chunk budget would be wrong"
        )
    return model


def load_reranker() -> CrossEncoder:
    """The pinned cross-encoder on CPU, truncating at its window."""
    _quiet()
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        RERANKER.name,
        revision=RERANKER.revision,
        device="cpu",
        max_length=RERANKER.max_tokens,
    )


def load_token_counter() -> TokenCounter:
    """Return a function counting the embedder's word pieces in a string,
    without the special tokens. Only the tokenizer is downloaded."""
    _quiet()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(EMBEDDER.name, revision=EMBEDDER.revision)
    tokenizer.model_max_length = 1 << 30  # counting only; never truncate or warn

    def count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    return count


def payload_budget() -> int:
    """Word pieces available to `path :: name\\ntext` inside the embedder window."""
    return EMBEDDER.max_tokens - SPECIAL_TOKENS
