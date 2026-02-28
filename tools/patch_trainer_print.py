#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Patch RootSAM_Pro/engine/trainer.py to print SoilFP / SoilLambda / SoilTopK every 5 steps.

- Only touches the terminal print block inside train_epoch (no training logic changes).
- Idempotent: if patch already applied, exits cleanly.
- Creates a backup: trainer.py.bak_print

Usage (from project root):
    python RootSAM_Pro/tools/patch_trainer_print.py
"""

from __future__ import annotations
import os
import re
from pathlib import Path


INSERT_BLOCK = """\
            # --- DPSL / Soil constraint telemetry (optional) ---
            if "SoilFP" in last_loss_dict:
                msg += f" | SoilFP: {last_loss_dict.get('SoilFP', 0.0):.4f}"
            if "SoilLambda" in last_loss_dict:
                msg += f" | SoilLam: {last_loss_dict.get('SoilLambda', 0.0):.4f}"
            if "SoilTopK" in last_loss_dict:
                try:
                    msg += f" | TopK: {int(last_loss_dict.get('SoilTopK', 0))}"
                except Exception:
                    msg += f" | TopK: {last_loss_dict.get('SoilTopK', 0)}"
"""


def main() -> int:
    root = Path(__file__).resolve().parents[1]  # RootSAM_Pro/
    trainer = root / "engine" / "trainer.py"
    if not trainer.exists():
        print(f"[ERROR] trainer.py not found at: {trainer}")
        return 2

    txt = trainer.read_text(encoding="utf-8", errors="replace")

    # Idempotency check
    if "DPSL / Soil constraint telemetry" in txt or "SoilLam" in txt:
        print("[OK] trainer.py already patched for Soil telemetry.")
        return 0

    # Find the specific print block inside train_epoch
    # We target the 'if step % 5 == 0 and last_loss_dict is not None:' block and insert before its 'print(msg)'.
    block_pat = re.compile(
        r"if\s+step\s*%\s*5\s*==\s*0\s+and\s+last_loss_dict\s+is\s+not\s+None\s*:\s*\n"
        r"(?P<body>(?:[ \t]+.*\n)+?)"
        r"(?P<printline>[ \t]+print\(msg\)\s*\n)",
        re.MULTILINE
    )

    m = block_pat.search(txt)
    if not m:
        # Fallback: insert before the first 'print(msg)' that is preceded by 'KeyframeGate' line
        fallback_pat = re.compile(r"(if\s+\"KeyframeGate\".*\n)([ \t]+print\(msg\)\s*\n)", re.MULTILINE)
        m2 = fallback_pat.search(txt)
        if not m2:
            print("[ERROR] Could not locate the train_epoch print(msg) block to patch.")
            return 3
        insert_at = m2.start(2)
        new_txt = txt[:insert_at] + INSERT_BLOCK + txt[insert_at:]
    else:
        insert_at = m.start("printline")
        new_txt = txt[:insert_at] + INSERT_BLOCK + txt[insert_at:]

    # Backup then write
    bak = trainer.with_suffix(".py.bak_print")
    bak.write_text(txt, encoding="utf-8")
    trainer.write_text(new_txt, encoding="utf-8")

    print(f"[OK] Patched: {trainer}")
    print(f"[OK] Backup : {bak}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
