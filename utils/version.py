"""RootSAM-Pro version utilities (no external deps)."""

from __future__ import annotations

__version__ = "2.0.0"
__codename__ = "OneShot-Modular-V2Final"

def make_run_version(code_hash_short: str | None = None) -> str:
    """Create a human-readable run version string.

    Example: 2.0.0+1a2b3c4d
    """
    if code_hash_short:
        code_hash_short = str(code_hash_short).strip()
    return f"{__version__}+{code_hash_short}" if code_hash_short else __version__
