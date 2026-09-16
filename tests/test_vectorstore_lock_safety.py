"""Lock files must never redirect persistence writes outside the store."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

from cheragh import Document, HashingEmbedding
from cheragh.vectorstores.memory import MemoryVectorStore, _store_file_lock


pytestmark = pytest.mark.skipif(importlib.util.find_spec("fcntl") is None, reason="POSIX advisory locks")


@pytest.mark.parametrize("exclusive", [False, True])
@pytest.mark.parametrize("nofollow", [False, True])
def test_symlink_lock_is_rejected_without_changing_target(tmp_path, monkeypatch, exclusive, nofollow):
    if not nofollow:
        monkeypatch.setattr(os, "O_NOFOLLOW", 0, raising=False)
    outside = tmp_path / "outside.txt"
    outside.write_text("unrelated configuration")
    store = tmp_path / "store"
    store.mkdir()
    (store / ".store.lock").symlink_to(outside)

    with pytest.raises((OSError, ValueError)):
        with _store_file_lock(store, exclusive=exclusive):
            pytest.fail("a symlink was accepted as a store lock")
    assert outside.read_text() == "unrelated configuration"


@pytest.mark.parametrize("backend", ["memory", "faiss"])
def test_save_cannot_truncate_an_external_lock_target(tmp_path, backend):
    if backend == "faiss":
        pytest.importorskip("faiss")
        from cheragh.vectorstores.faiss import FaissVectorStore
        store = FaissVectorStore(HashingEmbedding(8))
    else:
        store = MemoryVectorStore(HashingEmbedding(8))
    store.add_documents([Document("cat", doc_id="a")])
    destination = tmp_path / "index"
    destination.mkdir()
    outside = tmp_path / "configuration.txt"
    outside.write_text("preserve this file")
    (destination / ".store.lock").symlink_to(outside)

    with pytest.raises((OSError, ValueError)):
        store.save(destination)
    assert outside.read_text() == "preserve this file"
    assert not (destination / "manifest.json").exists()


@pytest.mark.parametrize("exclusive", [False, True])
def test_dangling_symlink_does_not_create_an_external_file_without_nofollow(tmp_path, monkeypatch, exclusive):
    monkeypatch.setattr(os, "O_NOFOLLOW", 0, raising=False)
    outside = tmp_path / "outside.txt"
    store = tmp_path / "store"
    store.mkdir()
    (store / ".store.lock").symlink_to(outside)
    with pytest.raises((OSError, ValueError)):
        with _store_file_lock(store, exclusive=exclusive):
            pytest.fail("dangling symlink accepted")
    assert not outside.exists()


def test_replaced_lock_inode_is_rejected_and_descriptor_closed(tmp_path, monkeypatch):
    path = tmp_path / ".store.lock"
    path.write_text("original")
    real_open = os.open
    opened = []

    def replace_during_open(target, flags, mode=0o777):
        fd = real_open(target, flags, mode)
        if Path(target) == path:
            opened.append(fd)
            path.unlink()
            path.write_text("replacement must remain untouched")
        return fd

    monkeypatch.setattr(os, "open", replace_during_open)
    with pytest.raises(ValueError, match="same regular file"):
        with _store_file_lock(tmp_path):
            pytest.fail("a replaced lock inode was accepted")
    assert path.read_text() == "replacement must remain untouched"
    assert len(opened) == 1
    with pytest.raises(OSError):
        os.fstat(opened[0])


def test_shared_lock_opens_read_only_and_does_not_rewrite_contents(tmp_path, monkeypatch):
    path = tmp_path / ".store.lock"
    path.write_text("writer bookkeeping")
    path.chmod(0o400)
    real_open = os.open

    def require_read_only(target, flags, mode=0o777):
        if Path(target) == path:
            assert not flags & (os.O_WRONLY | os.O_RDWR | os.O_TRUNC)
        return real_open(target, flags, mode)

    monkeypatch.setattr(os, "open", require_read_only)
    with _store_file_lock(tmp_path, exclusive=False):
        pass
    assert path.read_text() == "writer bookkeeping"


def test_read_only_legacy_store_without_lock_remains_loadable(tmp_path, monkeypatch):
    store = MemoryVectorStore(HashingEmbedding(8))
    store.add_documents([Document("cat", doc_id="a")])
    store.save(tmp_path)
    lock = tmp_path / ".store.lock"
    lock.unlink()
    real_open = os.open

    def read_only_directory(target, flags, mode=0o777):
        if Path(target) == lock and flags & os.O_CREAT:
            raise PermissionError("read-only directory")
        return real_open(target, flags, mode)

    monkeypatch.setattr(os, "open", read_only_directory)
    restored = MemoryVectorStore.load(tmp_path, HashingEmbedding(8))
    assert [document.doc_id for document in restored.documents] == ["a"]
    assert not lock.exists()


def test_lock_still_serializes_independent_processes(tmp_path):
    script = """
import sys
from pathlib import Path
from cheragh.vectorstores.memory import _store_file_lock
try:
    with _store_file_lock(Path(sys.argv[1]), timeout=0.05):
        print('acquired')
except TimeoutError:
    print('blocked')
"""
    environment = dict(os.environ)
    source = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")

    def contender():
        return subprocess.run(
            [sys.executable, "-c", script, str(tmp_path)],
            env=environment, check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip()

    with _store_file_lock(tmp_path):
        assert contender() == "blocked"
    assert contender() == "acquired"
