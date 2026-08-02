"""Shared pytest fixtures.

The filesystem memory store writes to disk during research (a runtime artifact,
like the ``reports/`` directory). Redirect its default root to a per-test temp
directory so the suite never pollutes the repository working tree.
"""

import pytest

from src.agents.filesystem_memory import FilesystemMemory


@pytest.fixture(autouse=True)
def _isolate_memory_root(tmp_path, monkeypatch):
    """Point ``FilesystemMemory``'s default root at a temp dir for every test."""
    monkeypatch.setattr(
        FilesystemMemory, "DEFAULT_ROOT", str(tmp_path / "memory")
    )
