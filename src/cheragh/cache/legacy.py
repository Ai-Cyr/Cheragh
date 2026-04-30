"""Backward-compatible retriever persistence helpers.

These functions preserve the older v0.x pickle retriever cache API used by
retrievers such as HybridSearchRetriever. New applications should prefer
CacheBackend implementations and sign persistent pickle caches with HMAC.

Security note: ``load_cache`` uses pickle for backward compatibility. Only load
legacy cache files from trusted locations. Set ``allow_unsafe_pickle=False`` to
turn cache hits into safe misses instead of deserializing untrusted data.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import tempfile
from typing import Any, Dict, List, Optional

from ..base import Document, EmbeddingModel

SCHEMA_VERSION = 1


def hash_documents(documents: List[Document]) -> str:
    h = hashlib.sha256()
    for d in documents:
        h.update((d.doc_id or "").encode("utf-8"))
        h.update(b"\x00")
        h.update(d.content.encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()


def embedder_fingerprint(embedder: EmbeddingModel) -> str:
    if hasattr(embedder, "get_fingerprint") and callable(embedder.get_fingerprint):
        try:
            return str(embedder.get_fingerprint())
        except Exception:
            pass
    return embedder.__class__.__name__


def save_cache(
    path: str,
    retriever_class: str,
    content_hash: str,
    embedder_fp: str,
    state: Dict[str, Any],
    extra_fingerprint: str = "",
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "retriever_class": retriever_class,
        "content_hash": content_hash,
        "embedder_fingerprint": embedder_fp,
        "extra_fingerprint": extra_fingerprint,
        "state": state,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    dir_name = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".cache_", suffix=".tmp", dir=dir_name)
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def load_cache(
    path: str,
    expected_class: str,
    expected_content_hash: str,
    expected_embedder_fp: str,
    expected_extra_fp: str = "",
    *,
    allow_unsafe_pickle: bool = True,
) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    if not allow_unsafe_pickle:
        return None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    except (pickle.UnpicklingError, EOFError, OSError, AttributeError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != SCHEMA_VERSION:
        return None
    if payload.get("retriever_class") != expected_class:
        return None
    if payload.get("content_hash") != expected_content_hash:
        return None
    if payload.get("embedder_fingerprint") != expected_embedder_fp:
        return None
    if payload.get("extra_fingerprint", "") != expected_extra_fp:
        return None
    return payload.get("state")
