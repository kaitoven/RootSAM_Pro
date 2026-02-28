"""Run manifest & code fingerprinting utilities.

Goal: top-journal reproducibility with *argparse-over-cfg* workflow.
- Captures resolved cfg (after CLI overrides)
- Captures command line
- Captures environment (python/torch/cuda/gpu)
- Captures code fingerprint (sha256 over selected source files)
- Captures ablation/module switches & trainable params

No external dependencies.
"""

from __future__ import annotations

import os
import sys
import json
import platform
import socket
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional


def _iter_source_files(project_root: str, include_dirs: List[str]) -> List[str]:
    project_root = os.path.abspath(project_root)
    out: List[str] = []
    for d in include_dirs:
        root = os.path.join(project_root, d)
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            # prune
            dirnames[:] = [x for x in dirnames if x not in ["__pycache__", ".git", ".idea", ".vscode", "runs", "checkpoints", "checkpoints_out", "data"]]
            for fn in filenames:
                if fn.endswith(".py") or fn.endswith(".yaml") or fn.endswith(".yml") or fn.endswith(".txt") or fn.endswith(".md") or fn.endswith(".json"):
                    out.append(os.path.join(dirpath, fn))
    out = sorted(set(out))
    return out


def compute_code_fingerprint(project_root: str,
                             include_dirs: Optional[List[str]] = None) -> Dict[str, Any]:
    """Compute a deterministic SHA256 fingerprint of source/config files."""
    include_dirs = include_dirs or ["configs", "datasets", "engine", "models", "tools", "utils"]
    files = _iter_source_files(project_root, include_dirs)

    file_entries: List[Dict[str, str]] = []
    h_all = hashlib.sha256()

    for p in files:
        try:
            with open(p, "rb") as f:
                data = f.read()
            h = hashlib.sha256(data).hexdigest()
            rel = os.path.relpath(p, project_root).replace("\\", "/")
            file_entries.append({"path": rel, "sha256": h})
            # fold into global
            h_all.update(rel.encode("utf-8"))
            h_all.update(h.encode("utf-8"))
        except Exception:
            continue

    return {
        "sha256": h_all.hexdigest(),
        "file_count": int(len(file_entries)),
        "files": file_entries,
        "project_root": os.path.abspath(project_root),
        "include_dirs": include_dirs,
    }


def cfg_to_dict(cfg) -> Dict[str, Any]:
    """Best-effort serialization for Config objects."""
    out: Dict[str, Any] = {}
    for k in dir(cfg):
        if k.startswith("_"):
            continue
        try:
            v = getattr(cfg, k)
        except Exception:
            continue
        if callable(v):
            continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = list(v)
        elif isinstance(v, dict):
            # shallow
            out[k] = {str(kk): vv for kk, vv in v.items()}
        else:
            out[k] = repr(v)
    return out


def _torch_env_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        import torch
        info["torch_version"] = getattr(torch, "__version__", "")
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        info["cudnn_version"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        if torch.cuda.is_available():
            info["cuda_device_count"] = int(torch.cuda.device_count())
            gpus = []
            for i in range(torch.cuda.device_count()):
                try:
                    p = torch.cuda.get_device_properties(i)
                    gpus.append({
                        "index": i,
                        "name": p.name,
                        "total_memory_GB": round(p.total_memory / (1024**3), 3),
                        "sm_count": getattr(p, "multi_processor_count", None),
                    })
                except Exception:
                    pass
            info["gpus"] = gpus
    except Exception:
        pass
    return info


def write_run_manifest(report_dir: str,
                       args,
                       cfg,
                       code_fingerprint: Dict[str, Any],
                       model=None,
                       criterion=None,
                       extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Write run manifest JSON files into report_dir."""
    os.makedirs(report_dir, exist_ok=True)
    extra = extra or {}

    # Args (argparse Namespace)
    args_dict = {}
    try:
        args_dict = vars(args)
    except Exception:
        try:
            args_dict = dict(args)
        except Exception:
            args_dict = {"_repr": repr(args)}

    # Trainable params + module toggles (auditable)
    trainable_model = None
    trainable_crit = None
    modules = {}
    try:
        if model is not None:
            trainable_model = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
            modules = {
                "hr_srd_washers": (getattr(model, "srd_washer_g0_raw", None) is not None),
                "kmr_adapter": (getattr(model, "kmr", None) is not None),
                "bhfi": (getattr(model, "bhfi", None) is not None),
                "bkmc": True,
                "learned_router": (getattr(model, "router", None) is not None),
                "dual_memory_bank": True,
            }
    except Exception:
        pass
    try:
        if criterion is not None:
            trainable_crit = int(sum(p.numel() for p in criterion.parameters() if p.requires_grad))
    except Exception:
        pass

    # Core manifest
    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "timestamp_local": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "cmdline": str(getattr(cfg, "CMDLINE", " ".join(sys.argv))),
        "run": {
            "run_id": str(getattr(cfg, "RUN_ID", "")),
            "run_name": str(getattr(cfg, "RUN_NAME", "")),
            "run_version": str(getattr(cfg, "RUN_VERSION", "")),
            "subset": str(getattr(cfg, "SUBSET_NAME", "")),
            "ablation_mode": str(getattr(cfg, "ABLATION_MODE", "")),
        },
        "paths": {
            "run_dir": str(getattr(cfg, "RUN_DIR", "")),
            "log_dir": str(getattr(cfg, "LOG_DIR", "")),
            "viz_dir": str(getattr(cfg, "VIZ_DIR", "")),
            "report_dir": str(getattr(cfg, "REPORT_DIR", report_dir)),
            "ckpt_dir": str(getattr(cfg, "CKPT_DIR", "")),
            "best_ckpt_path": str(getattr(cfg, "BEST_CKPT_PATH", "")),
        },
        "args": args_dict,
        "cfg_resolved": cfg_to_dict(cfg),
        "env": {
            "torch": _torch_env_info(),
        },
        "code_fingerprint": code_fingerprint,
        "trainable_params": {
            "model": trainable_model,
            "criterion": trainable_crit,
        },
        "modules": modules,
    }

    # Attach any extra info
    if extra:
        manifest["extra"] = extra

    # Write manifest
    manifest_path = os.path.join(report_dir, "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Backward-compatible ablation_manifest.json (kept small)
    ablation_manifest = {
        "subset": manifest["run"]["subset"],
        "ablation_mode": manifest["run"]["ablation_mode"],
        "run_id": manifest["run"]["run_id"],
        "run_version": manifest["run"]["run_version"],
        "code_hash_short": str(code_fingerprint.get("sha256", ""))[:8],
        "modules": modules,
        "trainable_params": manifest["trainable_params"],
    }
    with open(os.path.join(report_dir, "ablation_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(ablation_manifest, f, indent=2, ensure_ascii=False)

    # Minimal version marker
    try:
        with open(os.path.join(report_dir, "VERSION.txt"), "w", encoding="utf-8") as f:
            f.write(f"{manifest['run']['run_version']}\n")
            f.write(f"run_id={manifest['run']['run_id']}\n")
            f.write(f"code_sha256={code_fingerprint.get('sha256','')}\n")
    except Exception:
        pass

    return manifest
