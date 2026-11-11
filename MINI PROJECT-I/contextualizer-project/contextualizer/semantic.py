"""Optional dense-embedding backend (``pip install contextualizer[semantic]``)."""
from __future__ import annotations

from typing import Callable, Sequence


def load_sentence_transformer(model_name: str = "all-MiniLM-L6-v2") -> Callable[[Sequence[str]], list]:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Semantic mode needs sentence-transformers: pip install 'contextualizer[semantic]'"
        ) from exc
    model = SentenceTransformer(model_name)

    def embed(texts: Sequence[str]):
        return model.encode(list(texts), normalize_embeddings=True).tolist()

    return embed
