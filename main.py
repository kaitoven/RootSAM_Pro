import os
import cv2
import csv
import json
import sys
import platform
import socket
import hashlib
import re
import shlex
from datetime import datetime
import torch
import torch.nn.functional as F


def _auto_data_root() -> str:
    """Best-effort detect AutoDL data disk mount. Default: /root/autodl-tmp if writable."""
    cand = "/root/autodl-tmp"
    try:
        if os.path.isdir(cand) and os.access(cand, os.W_OK):
            return cand
    except Exception:
        pass
    return os.getcwd()


def _auto_run_root() -> str:
    """Default run_root on AutoDL: put runs on data disk to avoid 30GB system disk."""
    data_root = _auto_data_root()
    if os.path.abspath(data_root) == "/root/autodl-tmp":
        return os.path.join(data_root, "runs_rootsam_pro")
    return "runs"


def _auto_cache_root() -> str:
    """Default cache root on AutoDL data disk."""
    data_root = _auto_data_root()
    if os.path.abspath(data_root) == "/root/autodl-tmp":
        return os.path.join(data_root, ".cache")
    return ""


def setup_cache_env(cache_root: str, override: bool = False) -> None:
    """Redirect heavy caches (torch/hf) onto data disk to avoid system disk overflow.

    If override=False, existing env vars are respected.
    """
    if not cache_root:
        return
    try:
        os.makedirs(cache_root, exist_ok=True)
    except Exception:
        return

    def _set(k: str, v: str):
        if override or (k not in os.environ):
            os.environ[k] = v

    _set("XDG_CACHE_HOME", cache_root)
    _set("TORCH_HOME", os.path.join(cache_root, "torch"))
    _set("HF_HOME", os.path.join(cache_root, "hf"))
    _set("TRANSFORMERS_CACHE", os.path.join(cache_root, "hf"))
    for p in [os.environ["TORCH_HOME"], os.environ["HF_HOME"]]:
        try:
            os.makedirs(p, exist_ok=True)
        except Exception:
            pass

def _safe_scalar_from_modules(modules, attr_name, transform_fn, default=float("nan")):
    """Safely extract a scalar from the first module that has `attr_name`.

    This keeps main.py robust to modular refactors (e.g., router params moved off RootSAMPro).
    """
    for mod in modules:
        if mod is None:
            continue
        if hasattr(mod, attr_name):
            try:
                v = transform_fn(getattr(mod, attr_name))
                if torch.is_tensor(v):
                    v = v.detach().float().cpu().item()
                return float(v)
            except Exception:
                continue
    return float(default)


def _get_key_temp(model):
    # Router may live under the ASTA adapter after modular refactors.
    router = getattr(model, "router", None)
    asta = getattr(model, "asta", None)
    router2 = getattr(asta, "router", None) if asta is not None else None
    return _safe_scalar_from_modules(
        (router2, router, model),
        "key_temp_raw",
        lambda t: F.softplus(t).clamp(min=1e-6),
    )
import random
import argparse
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from configs.root_sam_pro_cfg import Config
from datasets.dataset_prmi import PRMI_KinematicDataset, ExtremeCurriculumSampler, GroupBySeqLenBatchSampler
from models.root_sam_pro import RootSAMPro
from engine.losses import TACEOptimizationField
from engine.trainer import train_epoch, evaluate_epoch
from utils.helpers import set_absolute_seed, LoggerEngine
from utils.version import __version__ as ROOTSAMPRO_VERSION, __codename__ as ROOTSAMPRO_CODENAME, make_run_version
from utils.manifest import compute_code_fingerprint, write_run_manifest, cfg_to_dict
from utils.optim import build_adamw_param_groups, build_adamw_param_groups_dual_lr

# Polar spectral SRD statistics (PP-FSRD++). Used only for logging, does not change training.
try:
    from models.adapters.srd_adapter import FourierWashingUnit
except Exception:
    FourierWashingUnit = None

# 🚨 【多进程护盾】：彻底关闭 OpenCV 多线程，防止与 DataLoader 多进程环境发生死锁风暴！
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def worker_init_fn(worker_id):
    """打破多进程 Fork 带来的随机种子同质化，保障数据增强的多样性"""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)


def _dl_perf_kwargs(num_workers: int, cfg):
    """Extra DataLoader kwargs for throughput (safe with num_workers==0)."""
    kw = {}
    try:
        nw = int(num_workers)
    except Exception:
        nw = 0
    if nw > 0:
        kw["persistent_workers"] = bool(getattr(cfg, "PERSISTENT_WORKERS", True))
        pf = int(getattr(cfg, "PREFETCH_FACTOR", 2))
        if pf < 1:
            pf = 2
        kw["prefetch_factor"] = pf
    return kw
    random.seed(worker_seed)


def _sanitize_tag(tag: str) -> str:
    if tag is None:
        return ""
    keep = []
    for ch in str(tag):
        if ch.isalnum() or ch in ['-', '_', '.']:
            keep.append(ch)
        else:
            keep.append('_')
    out = ''.join(keep).strip('_')
    return out[:64]


def _infer_scalar(v: str):
    """Infer a Python scalar from string (int/float/str). Does *not* coerce 0/1 into bool."""
    s = str(v).strip()
    # int
    if re.match(r"^[+-]?\d+$", s):
        try:
            return int(s)
        except Exception:
            pass
    # float (including scientific)
    sl = s.lower()
    if re.match(r"^[+-]?\d*\.\d+(e[+-]?\d+)?$", sl) or re.match(r"^[+-]?\d+e[+-]?\d+$", sl):
        try:
            return float(s)
        except Exception:
            pass
    return s


_BOOL_TRUE = {"true", "yes", "y", "on", "1"}
_BOOL_FALSE = {"false", "no", "n", "off", "0"}


def _coerce_by_type(base, raw: str):
    """Coerce raw string into the type of `base` when possible.

    This makes --set safe and reproducible: e.g., SEED=0 stays int(0) (not bool False).
    """
    s = str(raw).strip()
    # None -> infer
    if base is None:
        sl = s.lower()
        if sl in _BOOL_TRUE:
            return True
        if sl in _BOOL_FALSE:
            return False
        return _infer_scalar(s)

    # bool must be checked before int (since bool is a subclass of int)
    if isinstance(base, bool):
        sl = s.lower()
        if sl in _BOOL_TRUE:
            return True
        if sl in _BOOL_FALSE:
            return False
        # fallback: non-empty string -> True
        return bool(s)

    # ints
    if isinstance(base, int) and not isinstance(base, bool):
        return int(s)

    # floats
    if isinstance(base, float):
        return float(s)

    # strings
    if isinstance(base, str):
        return s

    # lists / tuples: accept JSON list or comma-separated
    if isinstance(base, (list, tuple)):
        sl = s.strip()
        try:
            if sl.startswith('[') and sl.endswith(']'):
                import json
                arr = json.loads(sl)
                return type(base)(arr)
        except Exception:
            pass
        parts = [pp.strip() for pp in sl.split(',') if pp.strip()]
        return type(base)(parts)

    # dict: JSON object
    if isinstance(base, dict):
        try:
            import json
            return json.loads(s)
        except Exception:
            return base

    # fallback
    return _infer_scalar(s)


def apply_set_overrides(cfg, kv_list):
    """Apply --set KEY=VALUE overrides onto cfg (flat attributes).

    - If KEY exists in cfg, cast VALUE to the existing type (bool/int/float/list/etc.).
    - Otherwise, infer a reasonable type.
    """
    if not kv_list:
        return
    for item in kv_list:
        if item is None:
            continue
        if '=' not in str(item):
            print(f"⚠️ Ignoring malformed --set '{item}'. Expected KEY=VALUE")
            continue
        k, v = str(item).split('=', 1)
        k = k.strip()
        if not k:
            continue
        base = getattr(cfg, k, None) if hasattr(cfg, k) else None
        try:
            val = _coerce_by_type(base, v)
        except Exception:
            # last resort
            val = _infer_scalar(v)
        setattr(cfg, k, val)


def discover_latest_run_id(run_root: str, subset: str, ablation: str):
    """Auto-detect latest RUN_ID under run_root for (subset, ablation). Returns '' if none."""
    try:
        if not os.path.isdir(run_root):
            return ""
        prefix = f"{subset}_{ablation}_"
        cands = []
        for name in os.listdir(run_root):
            full = os.path.join(run_root, name)
            if not os.path.isdir(full):
                continue
            if name.startswith(prefix):
                cands.append((os.path.getmtime(full), name))
        if not cands:
            return ""
        cands.sort(key=lambda x: x[0], reverse=True)
        latest_name = cands[0][1]
        return latest_name[len(prefix):]
    except Exception:
        return ""

def extract_trainable_state_dict(model):
    """🚨 PEFT 硬盘拯救机制：仅提取带有梯度的微调层权重保存，避免每次保存 6GB 的骨干网络炸毁硬盘"""
    trainable = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable[name] = param.detach().cpu()
    return trainable


def load_trainable_state_dict(model, ckpt_path, device):
    """安全加载微调权重，允许 strict=False 自动跳过被冻结的 SAM2 主干参数"""
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    print(f"✅ Successfully loaded physics adapters from {ckpt_path}")


def parse_args():
    """论文级消融控制台 + 动态覆盖 cfg（argparse 为主）"""
    parser = argparse.ArgumentParser(description="RootSAM-Pro Grand Unification Engine")

    # --- Core experiment identity ---
    parser.add_argument('--subset', type=str, default=None,
                        help='Target PRMI subset (e.g., Papaya_736x552_DPI150)')
    parser.add_argument('--ablation', type=str, default=None,
                        choices=["ZERO_SHOT","VANILLA","SFA_ONLY","SFA_ASTA","FULL","SRD_ONLY","SRD_KMR"],
                        help="Select Table 4.1 orthogonal ablation configuration")

    # --- Reproducible run directory & versioning ---
    parser.add_argument('--run_root', type=str, default=_auto_run_root(),
                    help='Root folder for run artifacts (default: auto on /root/autodl-tmp/runs_rootsam_pro when available)')
    parser.add_argument('--cache_root', type=str, default=_auto_cache_root(),
                    help='Cache root (torch/hf) to avoid filling system disk. Default: auto on /root/autodl-tmp/.cache if available.')
    parser.add_argument('--no_cache_redirect', action='store_true',
                    help='Disable auto cache redirection.')

    parser.add_argument('--run_id', type=str, default=None,
                        help='Explicit RUN_ID (for exact reproducibility / resuming a specific run)')
    parser.add_argument('--run_tag', type=str, default=None,
                        help='Human-friendly tag appended to auto RUN_ID (e.g., debugA, lr1e4)')
    parser.add_argument('--set', action='append', default=[],
                        help='Override cfg with KEY=VALUE (repeatable). Example: --set EPOCHS=10 --set LR=1e-4')
    parser.add_argument('--print_cfg', action='store_true',
                        help='Print resolved cfg after applying CLI overrides and exit')

    # --- Resume ---
    parser.add_argument('--resume', action='store_true', help='Resume from the latest checkpoint (auto-detect latest run if run_id not set)')

    return parser.parse_args()



def _csv_rewrite_with_new_header(csv_path: str, new_fieldnames: list):
    """当新增字段出现时，重写 CSV 以扩展表头（保持历史数据）。"""
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=new_fieldnames)
            writer.writeheader()
        return

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        old_rows = list(reader)
        old_fields = reader.fieldnames or []

    merged_fields = []
    seen = set()
    for k in list(old_fields) + list(new_fieldnames):
        if k not in seen:
            merged_fields.append(k)
            seen.add(k)

    tmp_path = csv_path + ".tmp"
    with open(tmp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=merged_fields)
        writer.writeheader()
        for r in old_rows:
            writer.writerow({k: r.get(k, "") for k in merged_fields})
    os.replace(tmp_path, csv_path)


def csv_append_row(csv_path: str, row: dict):
    """稳健写入：自动补全 header（支持运行中新增字段）。"""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    row = {str(k): row[k] for k in row.keys()}
    fieldnames = list(row.keys())

    if not os.path.exists(csv_path) or os.stat(csv_path).st_size == 0:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)
        return

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        existing_fields = reader.fieldnames or []

    need_expand = any(k not in existing_fields for k in fieldnames)
    if need_expand:
        _csv_rewrite_with_new_header(csv_path, existing_fields + fieldnames)

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        final_fields = reader.fieldnames or fieldnames

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=final_fields)
        writer.writerow({k: row.get(k, "") for k in final_fields})


def csv_write_single_row(csv_path: str, row: dict):
    """覆盖写入单行（用于 best_summary/test_summary）。"""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    row = {str(k): row[k] for k in row.keys()}
    fieldnames = list(row.keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def append_to_ablation_csv(cfg, test_res, ckpt_tag: str = "BEST_HPACS", hpacs_ok: int = 1, hpacs_reason: str = ""):
    """Append ONE row per ablation mode to the main Table 4.1 CSV.

    - If a strict-HPACS checkpoint exists, we write ckpt_tag=BEST_HPACS and hpacs_ok=1.
    - If no checkpoint satisfies the HPACS gate on VAL, we write ckpt_tag=BEST_SCORE (or LAST) and hpacs_ok=0.

    This keeps the ablation table to exactly 5 rows per subset (one per mode), while making feasibility explicit.
    """
    csv_file = "Table_4_1_Ablation_Results.csv"

    header = [
        "Subset", "Ablation_Mode", "CKPT_Tag", "HPACS_OK", "HPACS_Reason",
        "Global_mIoU", "Pure_Soil_FPR", "Insular_Recall", "Gap_mIoU", "clDice", "SDF_Relaxed_IoU",
    ]
    _csv_rewrite_with_new_header(csv_file, header)

    row = {
        "Subset": cfg.SUBSET_NAME,
        "Ablation_Mode": cfg.ABLATION_MODE,
        "CKPT_Tag": str(ckpt_tag),
        "HPACS_OK": int(hpacs_ok),
        "HPACS_Reason": str(hpacs_reason) if hpacs_reason is not None else "",
        "Global_mIoU": f"{test_res.get('Standard_IoU', 0):.2f}",
        "Pure_Soil_FPR": f"{test_res.get('Pure_Soil_FPR', 0):.2f}",
        "Insular_Recall": f"{test_res.get('Insular_Gap_Recall', 0):.2f}",
        "Gap_mIoU": f"{test_res.get('Gap_mIoU', 0):.2f}",
        "clDice": f"{test_res.get('clDice', 0):.2f}",
        "SDF_Relaxed_IoU": f"{test_res.get('SDF_Relaxed_IoU', 0):.2f}",
    }

    with open(csv_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writerow(row)

    print(f"📊 Quantitative Results securely appended to {csv_file}. Ready for LaTeX integration!")


def append_to_ablation_csv_all_ckpts(cfg, test_res, ckpt_tag: str):
    """Extended ablation CSV that records *which checkpoint* produced the numbers.

    Keeps Table_4_1_Ablation_Results.csv clean (5 rows), while preserving
    debuggability / rebuttal evidence in a separate file.
    """
    csv_file = "Table_4_1_Ablation_Results_AllCkpts.csv"

    header = [
        "Subset", "Ablation_Mode", "CKPT_Tag",
        "Global_mIoU", "Pure_Soil_FPR", "Insular_Recall", "Gap_mIoU", "clDice", "SDF_Relaxed_IoU",
    ]
    # Expand header safely if file already exists
    _csv_rewrite_with_new_header(csv_file, header)

    row = {
        "Subset": cfg.SUBSET_NAME,
        "Ablation_Mode": cfg.ABLATION_MODE,
        "CKPT_Tag": str(ckpt_tag),
        "Global_mIoU": f"{test_res.get('Standard_IoU', 0):.2f}",
        "Pure_Soil_FPR": f"{test_res.get('Pure_Soil_FPR', 0):.2f}",
        "Insular_Recall": f"{test_res.get('Insular_Gap_Recall', 0):.2f}",
        "Gap_mIoU": f"{test_res.get('Gap_mIoU', 0):.2f}",
        "clDice": f"{test_res.get('clDice', 0):.2f}",
        "SDF_Relaxed_IoU": f"{test_res.get('SDF_Relaxed_IoU', 0):.2f}",
    }
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writerow(row)



def collect_ppfsrd_polar_stats(model) -> dict:
    """Collect PP-FSRD++ (FourierWashingUnit) spectral shaping statistics.

    This is used purely for *auditable logging* (train_val_history.csv), proving that the
    frequency-domain SRD branch is learning a meaningful spectral policy.

    Returns an empty dict if FourierWashingUnit is unavailable or absent.
    """
    if FourierWashingUnit is None:
        return {}

    gammas = []
    w_means = []
    w_maxs = []
    rad_marginals = []
    ori_marginals = []

    for m in model.modules():
        if isinstance(m, FourierWashingUnit):
            try:
                g = m.gamma.detach().float().abs().mean().cpu()
                w = m.polar_band_w.detach().float().abs().cpu()  # (C,Kr,Ko)
                gammas.append(g)
                w_means.append(w.mean())
                w_maxs.append(w.max())
                rad_marginals.append(w.mean(dim=(0, 2)))  # (Kr,)
                ori_marginals.append(w.mean(dim=(0, 1)))  # (Ko,)
            except Exception:
                continue

    if len(gammas) == 0:
        return {"PPFSRD_num_washers": 0}

    out = {
        "PPFSRD_num_washers": int(len(gammas)),
        "PPFSRD_gamma_abs_mean": float(torch.stack(gammas).mean().item()),
        "PPFSRD_polar_w_abs_mean": float(torch.stack(w_means).mean().item()),
        "PPFSRD_polar_w_abs_max": float(torch.stack(w_maxs).max().item()),
    }

    # Optional marginals: which radial/orientation bands become salient.
    try:
        rad = torch.stack(rad_marginals, dim=0).mean(dim=0)  # (Kr,)
        ori = torch.stack(ori_marginals, dim=0).mean(dim=0)  # (Ko,)
        for i in range(int(rad.numel())):
            out[f"PPFSRD_rad_band_abs_mean_k{i}"] = float(rad[i].item())
        for i in range(int(ori.numel())):
            out[f"PPFSRD_ori_band_abs_mean_k{i}"] = float(ori[i].item())
    except Exception:
        pass

    return out





def main():
    args = parse_args()

    # -------------------------------------------------------------
    # AutoDL disk safety: redirect heavy caches (torch/hf) onto data disk
    # to avoid filling the 30GB system disk. Can be disabled via --no_cache_redirect.
    # -------------------------------------------------------------
    cache_root = getattr(args, "cache_root", None)
    if cache_root is None:
        cache_root = _auto_cache_root()
    if (not bool(getattr(args, "no_cache_redirect", False))) and cache_root:
        setup_cache_env(cache_root, override=False)

    cfg = Config()

    # -------------------------------------------------------------
    # CLI → cfg (argparse 传参为主，动态覆盖 cfg)
    # -------------------------------------------------------------
    if args.subset:
        cfg.SUBSET_NAME = args.subset
    if args.ablation:
        cfg.ABLATION_MODE = args.ablation
    apply_set_overrides(cfg, getattr(args, "set", []))

    cfg.RUN_ROOT = getattr(args, "run_root", "runs")
    cfg.CACHE_ROOT = cache_root
    cfg.CACHE_REDIRECT = (not bool(getattr(args, "no_cache_redirect", False))) and bool(cache_root)
    cfg.AUTO_DATA_ROOT = _auto_data_root()

    # -------------------------------------------------------------
    # Versioning & code fingerprint (no git required)
    # -------------------------------------------------------------
    project_root = os.path.dirname(os.path.abspath(__file__))
    code_fp = compute_code_fingerprint(project_root)
    code_sha = str(code_fp.get("sha256", ""))
    code_short = code_sha[:8] if code_sha else "unknown"

    # Resolve RUN_ID (deterministic + resume-friendly)
    run_id = (getattr(args, "run_id", None) or str(getattr(cfg, "RUN_ID", "")).strip() or "")
    if bool(getattr(args, "resume", False)) and (not run_id):
        detected = discover_latest_run_id(cfg.RUN_ROOT, cfg.SUBSET_NAME, cfg.ABLATION_MODE)
        if detected:
            run_id = detected
            print(f"🔁 Auto-detected latest run_id for resume: {run_id}")

    if not run_id:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = _sanitize_tag(getattr(args, "run_tag", None))
        run_id = f"{ts}_{code_short}" + (f"_{tag}" if tag else "")

    cfg.RUN_ID = run_id
    cfg.CODE_HASH_SHA256 = code_sha
    cfg.RUN_VERSION = make_run_version(code_short)
    cfg.CMDLINE = ' '.join(shlex.quote(x) for x in sys.argv)
    cfg.HOSTNAME = socket.gethostname()
    cfg.PLATFORM = platform.platform()

    cfg.setup_dirs()

    if bool(getattr(args, "print_cfg", False)):
        print(json.dumps(cfg_to_dict(cfg), indent=2, ensure_ascii=False))
        return

    set_absolute_seed(cfg.SEED)

    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')
    logger = LoggerEngine(log_dir=cfg.LOG_DIR)

    print("\n" + "=" * 70)
    print(f"🚀 RootSAM-Pro Grand Unification Engine Booting...")
    print(f"🏷️  Run ID       : {cfg.RUN_ID}")
    print(f"🧬 Version      : {cfg.RUN_VERSION} ({ROOTSAMPRO_CODENAME})")
    print(f"🧾 Code Hash    : {code_short}")
    print(f"🎯 Target Subset : {cfg.SUBSET_NAME}")
    print(f"🎛️ Ablation Mode : {cfg.ABLATION_MODE}")
    print(f"💻 Compute Node  : {device}")
    print("=" * 70 + "\n")
    g = torch.Generator()
    g.manual_seed(cfg.SEED)

    train_mode = str(getattr(cfg, 'TRAIN_MODE', 'CLIP')).upper()
    train_max_seq_len = int(getattr(cfg, 'TRAIN_MAX_SEQ_LEN', 0) or 0)
    ds_train = PRMI_KinematicDataset(cfg.ROOT_DIR, cfg.SUBSET_NAME, split='train', seq_length=cfg.SEQ_LENGTH,
                                     target_size=cfg.TARGET_SIZE, train_mode=train_mode, train_max_seq_len=train_max_seq_len)
    if train_mode == 'SEQUENCE':
        # Full-sequence rollout training: must batch by exact seq_len to avoid temporal misalignment.
        sampler = GroupBySeqLenBatchSampler(ds_train, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True)
        train_loader = DataLoader(
            ds_train,
            batch_sampler=sampler,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=bool(getattr(cfg, 'PIN_MEMORY', True)),
            worker_init_fn=worker_init_fn,
            generator=g,
            **_dl_perf_kwargs(cfg.NUM_WORKERS, cfg),
        )
    else:
        # Clip training (fast): extreme curriculum sampler over fixed-length snippets.
        sampler = ExtremeCurriculumSampler(ds_train, batch_size=cfg.BATCH_SIZE)
        train_loader = DataLoader(
            ds_train,
            batch_sampler=sampler,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=bool(getattr(cfg, 'PIN_MEMORY', True)),
            worker_init_fn=worker_init_fn,
            generator=g,
            **_dl_perf_kwargs(cfg.NUM_WORKERS, cfg),
        )

    ds_val = PRMI_KinematicDataset(cfg.ROOT_DIR, cfg.SUBSET_NAME, split='val', seq_length=cfg.SEQ_LENGTH,
                                   target_size=cfg.TARGET_SIZE)
    eval_bs = int(getattr(cfg, "EVAL_BATCH_SIZE", 1))
    val_sampler = GroupBySeqLenBatchSampler(ds_val, batch_size=eval_bs, shuffle=False, drop_last=False)
    val_loader = DataLoader(
        ds_val,
        batch_sampler=val_sampler,
        num_workers=4,
        pin_memory=bool(getattr(cfg, "PIN_MEMORY", True)),
        **_dl_perf_kwargs(4, cfg),
    )

    ds_test = PRMI_KinematicDataset(cfg.ROOT_DIR, cfg.SUBSET_NAME, split='test', seq_length=cfg.SEQ_LENGTH,
                                    target_size=cfg.TARGET_SIZE)
    test_sampler = GroupBySeqLenBatchSampler(ds_test, batch_size=eval_bs, shuffle=False, drop_last=False)
    test_loader = DataLoader(
        ds_test,
        batch_sampler=test_sampler,
        num_workers=4,
        pin_memory=bool(getattr(cfg, "PIN_MEMORY", True)),
        **_dl_perf_kwargs(4, cfg),
    )

    model = RootSAMPro(cfg).to(device)


    # -------------------------------------------------------------
    # Loss / Optimization Field (may carry learnable parameters such as
    # homoscedastic log-variances and dual variables). Must be created
    # before manifest/optimizer so it can be audited and optimized.
    # -------------------------------------------------------------
    try:
        criterion = TACEOptimizationField(cfg, device=device).to(device)
    except TypeError:
        criterion = TACEOptimizationField(cfg).to(device)

    # -------------------------------------------------------------
    # Reproducibility: dump a full run manifest (args + resolved cfg + env + code hash).
    # This makes Table 4.1 auditable from saved artifacts and prevents
    # "feature smuggling" accusations during review.
    # -------------------------------------------------------------
    try:
        write_run_manifest(
            report_dir=cfg.REPORT_DIR,
            args=args,
            cfg=cfg,
            code_fingerprint=code_fp,
            model=model,
            criterion=criterion,
            extra={
                "device": str(device),
                "torch_cuda_available": bool(torch.cuda.is_available()),
                "torch_cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
                "io": {
                    "data_root": str(getattr(cfg, "AUTO_DATA_ROOT", "")),
                    "run_root": str(getattr(cfg, "RUN_ROOT", "")),
                    "cache_root": str(getattr(cfg, "CACHE_ROOT", "")),
                    "cache_redirect": bool(getattr(cfg, "CACHE_REDIRECT", False)),
                    "cache_env": {
                        "XDG_CACHE_HOME": os.environ.get("XDG_CACHE_HOME", ""),
                        "TORCH_HOME": os.environ.get("TORCH_HOME", ""),
                        "HF_HOME": os.environ.get("HF_HOME", ""),
                        "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE", ""),
                    },
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                },
            },
        )
    except Exception as e:
        print(f"⚠️ Failed to write run manifest: {e}")

    # IMPORTANT: criterion may carry learnable parameters (e.g., homoscedastic log-variances)
    trainable_params = [p for p in model.parameters() if p.requires_grad] + [
        p for p in criterion.parameters() if p.requires_grad
    ]

    history_csv = os.path.join(cfg.REPORT_DIR, "train_val_history.csv")
    best_events_csv = os.path.join(cfg.REPORT_DIR, "best_events.csv")
    best_summary_csv = os.path.join(cfg.REPORT_DIR, "best_summary.csv")
    test_summary_csv = os.path.join(cfg.REPORT_DIR, "test_summary.csv")
    test_runs_csv = os.path.join(cfg.REPORT_DIR, "test_runs.csv")

    if cfg.ABLATION_MODE == "ZERO_SHOT" or len(trainable_params) == 0:
        print("❄️ ZERO-SHOT MODE ACTIVATED: Bypassing Training Pipeline...")
        test_res = evaluate_epoch(model, test_loader, device, cfg, output_viz_dir=cfg.VIZ_DIR, report_dir=cfg.REPORT_DIR)

        test_row = {"phase": "test", "run_id": getattr(cfg, "RUN_ID", ""), "run_version": getattr(cfg, "RUN_VERSION", ""), "code_hash": str(getattr(cfg, "CODE_HASH_SHA256", ""))[:8], "subset": cfg.SUBSET_NAME, "ablation": cfg.ABLATION_MODE}
        test_row.update({k: test_res.get(k, "") for k in test_res.keys()})
        # PP-FSRD++ spectral statistics (if present)
        try:
            test_row.update(collect_ppfsrd_polar_stats(model))
        except Exception:
            pass
        csv_write_single_row(test_summary_csv, test_row)
        csv_append_row(test_runs_csv, test_row)

        append_to_ablation_csv(cfg, test_res)
        append_to_ablation_csv_all_ckpts(cfg, test_res, ckpt_tag="ZERO_SHOT")
        return

    # --- Production optimizer: parameter-wise weight decay (safe for long runs) ---
    _named = []
    for _n, _p in model.named_parameters():
        if _p.requires_grad:
            _named.append((f"model.{_n}", _p))
    for _n, _p in criterion.named_parameters():
        if _p.requires_grad:
            _named.append((f"criterion.{_n}", _p))
    dual_lr_mult = float(getattr(cfg, "SOIL_DUAL_LR_MULT", 10.0))
    _param_groups = build_adamw_param_groups_dual_lr(
        _named,
        weight_decay=cfg.WEIGHT_DECAY,
        base_lr=cfg.LR,
        dual_lr_mult=dual_lr_mult,
    )
    optimizer = torch.optim.AdamW(_param_groups)# Warmup epochs (linear ramp MIN_LR -> LR), then cosine anneal to MIN_LR.
    warmup_epochs = int(getattr(cfg, "WARMUP_EPOCHS", 0))
    if warmup_epochs < 0:
        warmup_epochs = 0
    if cfg.EPOCHS <= 1:
        warmup_epochs = 0
    warmup_epochs = min(warmup_epochs, max(0, cfg.EPOCHS - 1))

    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, cfg.EPOCHS - warmup_epochs), eta_min=cfg.MIN_LR)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == 'cuda'))

    # -------------------------------
    # Dual-best checkpointing to eliminate "checkpoint selection bias"
    #   - BEST_HPACS: best under strict HPACS gate (clean 5-row Table 4.1)
    #   - BEST_SCORE: best by score without gate (sanity/rebuttal/ablation debugging)
    #   - LAST: last epoch weights (failsafe)
    # -------------------------------
    best_score_hpacs = 0.0
    best_epoch_hpacs = 0
    best_score_unconstrained = 0.0
    best_epoch_unconstrained = 0

    # Backward-compatible aliases (existing code expects best_score/best_epoch)
    best_score = best_score_hpacs
    best_epoch = best_epoch_hpacs

    start_epoch = 0
    ckpt_prefix = f"rootsam_pro_{getattr(cfg, 'RUN_NAME', f'{cfg.SUBSET_NAME}_{cfg.ABLATION_MODE}')}"
    latest_ckpt_path = f"{cfg.CKPT_DIR}/{ckpt_prefix}_latest.pth"

    # trainable-only ckpt paths (PEFT-safe, disk-friendly)
    best_hpacs_ckpt_path = cfg.BEST_CKPT_PATH  # backward-compatible "best.pth" semantics
    best_hpacs_ckpt_path_explicit = f"{cfg.CKPT_DIR}/{ckpt_prefix}_best_hpacs.pth"
    best_score_ckpt_path = f"{cfg.CKPT_DIR}/{ckpt_prefix}_best_score.pth"
    last_trainable_ckpt_path = f"{cfg.CKPT_DIR}/{ckpt_prefix}_last.pth"

    if args.resume and os.path.exists(latest_ckpt_path):
        print(f"🔄 Resuming training from {latest_ckpt_path}...")
        ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)

        model.load_state_dict(ckpt.get('model_state_dict', {}), strict=False)
        try:
            if 'criterion_state_dict' in ckpt:
                criterion.load_state_dict(ckpt.get('criterion_state_dict', {}), strict=False)
        except Exception as e:
            print(f"⚠️ Criterion state restore skipped: {e}")

        start_epoch = int(ckpt.get('epoch', 0))
        # Restore dual-best bookkeeping (fall back to legacy keys)
        best_score_hpacs = float(ckpt.get('best_score_hpacs', ckpt.get('best_score', 0.0)))
        best_epoch_hpacs = int(ckpt.get('best_epoch_hpacs', ckpt.get('best_epoch', 0)))
        best_score_unconstrained = float(ckpt.get('best_score_unconstrained', 0.0))
        best_epoch_unconstrained = int(ckpt.get('best_epoch_unconstrained', 0))

        best_score = best_score_hpacs
        best_epoch = best_epoch_hpacs

        cfg._BEST_FPR = float(ckpt.get('best_fpr', 100.0))
        cfg._BEST_INS = float(ckpt.get('best_ins', 0.0))
        cfg._BEST_SDF = float(ckpt.get('best_sdf', 0.0))

        cfg._BEST_CLD = float(ckpt.get('best_cld', 0.0))

        cfg._BEST_SCORE_FPR = float(ckpt.get('best_score_fpr', 100.0))
        cfg._BEST_SCORE_INS = float(ckpt.get('best_score_ins', 0.0))
        cfg._BEST_SCORE_SDF = float(ckpt.get('best_score_sdf', 0.0))
        cfg._BEST_SCORE_CLD = float(ckpt.get('best_score_cld', 0.0))
        try:
            optimizer.load_state_dict(ckpt.get('optimizer_state_dict', {}))
            scheduler.load_state_dict(ckpt.get('scheduler_state_dict', {}))
            scaler.load_state_dict(ckpt.get('scaler_state_dict', {}))
            print("✅ Optimizer/Scheduler/Scaler states restored.")
        except Exception as e:
            print(f"⚠️ Optimizer/Scheduler/Scaler restore skipped due to mismatch: {e}")
            print("⚠️ Continuing with fresh optimizer states (safer than partial restore).")

        print(
            f"✅ Successfully resumed at Epoch {start_epoch + 1} | "
            f"best_hpacs={best_score_hpacs:.6f} (ep={best_epoch_hpacs}) | "
            f"best_score={best_score_unconstrained:.6f} (ep={best_epoch_unconstrained})"
        )

    print("\n💥 Starting Truncated-BPTT Kinematic Training...")
    for epoch in range(start_epoch, cfg.EPOCHS):
        # Warmup: linear ramp MIN_LR -> LR for the first warmup_epochs epochs
        if warmup_epochs > 0 and epoch < warmup_epochs:
            lr_w = cfg.MIN_LR + (cfg.LR - cfg.MIN_LR) * float(epoch + 1) / float(warmup_epochs)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_w
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n--- Epoch {epoch + 1}/{cfg.EPOCHS} | LR: {current_lr:.6e} ---")

        train_out = train_epoch(model, train_loader, optimizer, scaler, criterion, device, cfg)
        if isinstance(train_out, dict):
            train_loss = float(train_out.get('loss_total', 0.0))
        else:
            train_loss = float(train_out)
        # Cosine schedule after warmup
        if warmup_epochs <= 0 or epoch >= warmup_epochs:
            scheduler.step()

        logger.log(epoch + 1, {"Loss": train_loss, "LR": current_lr})

        val_res = evaluate_epoch(model, val_loader, device, cfg, output_viz_dir=None)
        logger.log(epoch + 1, val_res, phase="Val")

        f1 = float(val_res.get('F1_Score', 0.0))
        cld = float(val_res.get('clDice', 0.0))
        sdf = float(val_res.get('SDF_Relaxed_IoU', 0.0))
        fpr = float(val_res.get('Pure_Soil_FPR', 100.0))
        recall = float(val_res.get('Recall', 0.0))
        ins = float(val_res.get('Insular_Gap_Recall', 0.0))

        FPR_MAX = float(getattr(cfg, "FPR_MAX_FOR_BEST_PCT", getattr(cfg, "FPR_MAX_FOR_BEST", 5.0)))
        R_MIN = float(getattr(cfg, "RECALL_MIN_FOR_BEST_PCT", getattr(cfg, "RECALL_MIN_FOR_BEST", 10.0)))

        CLD_MIN = float(getattr(cfg, "CLDICE_MIN_FOR_BEST_PCT", getattr(cfg, "CLDICE_MIN_FOR_BEST", 0.0)))

        passed_gate_strict = (fpr <= FPR_MAX) and (recall >= R_MIN) and ((CLD_MIN <= 0.0) or (cld >= CLD_MIN))
        # Strict HPACS gate: no warmup/free-pass allowed (auditable constraints).
        passed_gate = passed_gate_strict

        f1n, cldn, sdfn = f1 / 100.0, cld / 100.0, sdf / 100.0
        tahs = (2.0 * f1n * cldn) / (f1n + cldn + 1e-6)
        LAMBDA_SDF = float(getattr(cfg, "LAMBDA_SDF_IN_BEST", 0.25))
        current_score = tahs * (1.0 + LAMBDA_SDF * sdfn)

        best_eps = float(getattr(cfg, "BEST_TIE_EPS", 5e-4))

        # (A) HPACS-best (strict gate)
        should_update_hpacs = False
        if passed_gate_strict:
            if current_score > best_score_hpacs + best_eps:
                should_update_hpacs = True
            elif abs(current_score - best_score_hpacs) <= best_eps:
                best_fpr = float(getattr(cfg, "_BEST_FPR", 100.0))
                best_cld = float(getattr(cfg, "_BEST_CLD", 0.0))
                best_ins = float(getattr(cfg, "_BEST_INS", 0.0))
                best_sdf = float(getattr(cfg, "_BEST_SDF", 0.0))

                if (fpr < best_fpr - 1e-6) or \
                   (abs(fpr - best_fpr) <= 1e-6 and cld > best_cld + 1e-6) or \
                   (abs(fpr - best_fpr) <= 1e-6 and abs(cld - best_cld) <= 1e-6 and ins > best_ins + 1e-6) or \
                   (abs(fpr - best_fpr) <= 1e-6 and abs(cld - best_cld) <= 1e-6 and abs(ins - best_ins) <= 1e-6 and sdf > best_sdf + 1e-6):
                    should_update_hpacs = True

        # (B) Score-best (no gate) — prevents "checkpoint selection bias" in short runs
        should_update_score = False
        if current_score > best_score_unconstrained + best_eps:
            should_update_score = True
        elif abs(current_score - best_score_unconstrained) <= best_eps:
            # tie-break: prefer lower FPR, then higher Insular recall, then higher SDF
            # (purely for stable ranking; does NOT impose HPACS constraints)
            best_fpr_u = float(getattr(cfg, "_BEST_SCORE_FPR", 100.0))
            best_cld_u = float(getattr(cfg, "_BEST_SCORE_CLD", 0.0))
            best_ins_u = float(getattr(cfg, "_BEST_SCORE_INS", 0.0))
            best_sdf_u = float(getattr(cfg, "_BEST_SCORE_SDF", 0.0))
            if (fpr < best_fpr_u - 1e-6) or \
               (abs(fpr - best_fpr_u) <= 1e-6 and cld > best_cld_u + 1e-6) or \
               (abs(fpr - best_fpr_u) <= 1e-6 and abs(cld - best_cld_u) <= 1e-6 and ins > best_ins_u + 1e-6) or \
               (abs(fpr - best_fpr_u) <= 1e-6 and abs(cld - best_cld_u) <= 1e-6 and abs(ins - best_ins_u) <= 1e-6 and sdf > best_sdf_u + 1e-6):
                should_update_score = True

        pra = getattr(model, "pra", None)
        history_row = {
            "epoch": epoch + 1,
            "run_id": getattr(cfg, "RUN_ID", ""),
            "run_version": getattr(cfg, "RUN_VERSION", ""),
            "code_hash": str(getattr(cfg, "CODE_HASH_SHA256", ""))[:8],
            "subset": cfg.SUBSET_NAME,
            "ablation": cfg.ABLATION_MODE,
            "lr": float(current_lr),
            "train_loss": float(train_loss),
            # train components (if available)
            "train_loss_total": float(train_out.get('loss_total', train_loss)) if isinstance(train_out, dict) else float(train_loss),
            "train_loss_seg": float(train_out.get('loss_seg', 0.0)) if isinstance(train_out, dict) else "",
            "train_loss_probe": float(train_out.get('loss_probe', 0.0)) if isinstance(train_out, dict) else "",
            "train_loss_kin": float(train_out.get('loss_kin', 0.0)) if isinstance(train_out, dict) else "",
            "train_loss_pres": float(train_out.get('loss_pres', 0.0)) if isinstance(train_out, dict) else "",
            "train_loss_gate": float(train_out.get('loss_gate', 0.0)) if isinstance(train_out, dict) else "",
            "hpacs_score": float(current_score),
            "hpacs_pass_gate": int(bool(passed_gate_strict)),
            # Robust to modular refactors (e.g., PRA params moved under model.pra)
            "PRES_TAU": _safe_scalar_from_modules((pra, model), "pres_tau_logit", lambda t: torch.sigmoid(t), default=float('nan')),
            "ABSENT_BIAS": _safe_scalar_from_modules((pra, model), "abs_bias_mag_raw", lambda t: (-F.softplus(t)), default=float('nan')),
            "FIREWALL_ETA": _safe_scalar_from_modules((pra, model), "eta_energy_raw", lambda t: F.softplus(t), default=float('nan')),
            "FIREWALL_THETA": _safe_scalar_from_modules((pra, model), "theta_frag_raw", lambda t: torch.sigmoid(t), default=float('nan')),
            "KEY_TEMP": _get_key_temp(model),
        }
        # PP-FSRD++ spectral shaping stats (auditable evidence)
        try:
            history_row.update(collect_ppfsrd_polar_stats(model))
        except Exception:
            pass
        history_row.update({k: val_res.get(k, "") for k in val_res.keys()})
        csv_append_row(history_csv, history_row)

        # Always write LAST trainable weights (failsafe, loadable by load_trainable_state_dict)
        torch.save(extract_trainable_state_dict(model), last_trainable_ckpt_path)

        if should_update_score:
            best_score_unconstrained = current_score
            best_epoch_unconstrained = epoch + 1
            cfg._BEST_SCORE_FPR = fpr
            cfg._BEST_SCORE_INS = ins
            cfg._BEST_SCORE_SDF = sdf
            cfg._BEST_SCORE_CLD = cld
            torch.save(extract_trainable_state_dict(model), best_score_ckpt_path)
            print(
                f"🌟 [New Best-Score] Saved! Epoch={best_epoch_unconstrained} | Score={best_score_unconstrained:.6f} (no gate) | "
                f"F1={f1:.2f}% clDice={cld:.2f}% SDF={sdf:.2f}% FPR={fpr:.2f}% INS={ins:.2f}%"
            )

        if should_update_hpacs:
            best_score_hpacs = current_score
            best_epoch_hpacs = epoch + 1
            cfg._BEST_FPR = fpr
            cfg._BEST_INS = ins
            cfg._BEST_SDF = sdf

            cfg._BEST_CLD = cld
            # Backward-compatible best.pth (HPACS-best)
            torch.save(extract_trainable_state_dict(model), best_hpacs_ckpt_path)
            # Explicit tag filename
            torch.save(extract_trainable_state_dict(model), best_hpacs_ckpt_path_explicit)

            best_score = best_score_hpacs
            best_epoch = best_epoch_hpacs

            print(f"🌟 [New Best-HPACS] Saved! Epoch={best_epoch_hpacs} | Score={best_score_hpacs:.6f} | "
                  f"F1={f1:.2f}% clDice={cld:.2f}% SDF={sdf:.2f}% FPR={fpr:.2f}% INS={ins:.2f}%")

            best_row = dict(history_row)
            best_row.update({
                "best_epoch": best_epoch_hpacs,
                "best_score": float(best_score_hpacs),
                "best_epoch_hpacs": best_epoch_hpacs,
                "best_score_hpacs": float(best_score_hpacs),
                "best_epoch_unconstrained": best_epoch_unconstrained,
                "best_score_unconstrained": float(best_score_unconstrained),
            })
            csv_append_row(best_events_csv, best_row)
            csv_write_single_row(best_summary_csv, best_row)
        else:
            if not passed_gate_strict:
                fail_reasons = []
                if fpr > FPR_MAX:
                    fail_reasons.append(f"FPR={fpr:.2f}%>{FPR_MAX:.2f}%")
                if recall < R_MIN:
                    fail_reasons.append(f"Recall={recall:.2f}%<{R_MIN:.2f}%")
                if len(fail_reasons) == 0:
                    fail_reasons.append("Unknown")
                print("⚠️ [Skip Best-HPACS] Gate failed: " + " & ".join(fail_reasons))

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': extract_trainable_state_dict(model),
            'criterion_state_dict': criterion.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            # legacy (HPACS-best)
            'best_epoch': best_epoch_hpacs,
            'best_score': best_score_hpacs,
            # dual-best
            'best_epoch_hpacs': best_epoch_hpacs,
            'best_score_hpacs': best_score_hpacs,
            'best_epoch_unconstrained': best_epoch_unconstrained,
            'best_score_unconstrained': best_score_unconstrained,
            'best_fpr': float(getattr(cfg, "_BEST_FPR", 100.0)),
            'best_ins': float(getattr(cfg, "_BEST_INS", 0.0)),
            'best_sdf': float(getattr(cfg, "_BEST_SDF", 0.0)),
            'best_cld': float(getattr(cfg, "_BEST_CLD", 0.0)),
            'best_score_fpr': float(getattr(cfg, "_BEST_SCORE_FPR", 100.0)),
            'best_score_ins': float(getattr(cfg, "_BEST_SCORE_INS", 0.0)),
            'best_score_sdf': float(getattr(cfg, "_BEST_SCORE_SDF", 0.0)),
            'best_score_cld': float(getattr(cfg, "_BEST_SCORE_CLD", 0.0)),
        }, latest_ckpt_path)

    print("\n" + "🔥" * 25)
    print("🚀 RUNNING FINAL EVALUATION ON OFFICIAL TEST SPLIT...")
    print("🔥" * 25)

    # --------------------------------------------
    # Final evaluation: run multiple checkpoints to avoid "checkpoint selection bias".
    #   - BEST_HPACS => used for clean 5-row Table 4.1
    #   - BEST_SCORE / LAST => recorded in an extended CSV for debugging & rebuttal
    # --------------------------------------------
    candidates = []
    if os.path.exists(best_hpacs_ckpt_path):
        candidates.append(("BEST_HPACS", best_hpacs_ckpt_path))
    if os.path.exists(best_score_ckpt_path):
        candidates.append(("BEST_SCORE", best_score_ckpt_path))
    if os.path.exists(last_trainable_ckpt_path):
        candidates.append(("LAST", last_trainable_ckpt_path))

    best_hpacs_test_res = None

    if len(candidates) == 0:
        print("⚠️ No checkpoint candidates found. Evaluating current in-memory weights.")
        test_res = evaluate_epoch(model, test_loader, device, cfg, output_viz_dir=cfg.VIZ_DIR, report_dir=cfg.REPORT_DIR)
        logger.log(cfg.EPOCHS, test_res, phase="Test_Final:CURRENT")
        append_to_ablation_csv(cfg, test_res)
        append_to_ablation_csv_all_ckpts(cfg, test_res, ckpt_tag="CURRENT")
        print("\n🎉 All processes completed successfully!")
        print(f"📁 CSV saved under: {cfg.REPORT_DIR}")
        return

    for tag, ckpt_path in candidates:
        load_trainable_state_dict(model, ckpt_path, device)

        # Avoid overwriting the main report; keep BEST_HPACS in root report_dir for compatibility.
        tag_report_dir = cfg.REPORT_DIR if tag == "BEST_HPACS" else os.path.join(cfg.REPORT_DIR, f"test_{tag.lower()}")
        tag_viz_dir = cfg.VIZ_DIR if tag == "BEST_HPACS" else None

        test_res = evaluate_epoch(model, test_loader, device, cfg, output_viz_dir=tag_viz_dir, report_dir=tag_report_dir)
        logger.log(cfg.EPOCHS, test_res, phase=f"Test_Final:{tag}")

        pra = getattr(model, "pra", None)
        test_row = {
            "phase": "test",
            "run_id": getattr(cfg, "RUN_ID", ""),
            "run_version": getattr(cfg, "RUN_VERSION", ""),
            "code_hash": str(getattr(cfg, "CODE_HASH_SHA256", ""))[:8],
            "ckpt_tag": tag,
            "subset": cfg.SUBSET_NAME,
            "ablation": cfg.ABLATION_MODE,
            "best_epoch_hpacs": best_epoch_hpacs,
            "best_score_hpacs": float(best_score_hpacs),
            "best_epoch_unconstrained": best_epoch_unconstrained,
            "best_score_unconstrained": float(best_score_unconstrained),
            # Robust to modular refactors (e.g., PRA params moved under model.pra)
            "PRES_TAU": _safe_scalar_from_modules((pra, model), "pres_tau_logit", lambda t: torch.sigmoid(t), default=float('nan')),
            "ABSENT_BIAS": _safe_scalar_from_modules((pra, model), "abs_bias_mag_raw", lambda t: (-F.softplus(t)), default=float('nan')),
            "FIREWALL_ETA": _safe_scalar_from_modules((pra, model), "eta_energy_raw", lambda t: F.softplus(t), default=float('nan')),
            "FIREWALL_THETA": _safe_scalar_from_modules((pra, model), "theta_frag_raw", lambda t: torch.sigmoid(t), default=float('nan')),
            "KEY_TEMP": _get_key_temp(model),
        }
        # PP-FSRD++ spectral statistics (if present)
        try:
            test_row.update(collect_ppfsrd_polar_stats(model))
        except Exception:
            pass
        test_row.update({k: test_res.get(k, "") for k in test_res.keys()})

        # Write per-tag test summary (does not overwrite BEST_HPACS summary)
        tag_summary_path = test_summary_csv if tag == "BEST_HPACS" else os.path.join(tag_report_dir, "test_summary.csv")
        csv_write_single_row(tag_summary_path, test_row)
        csv_append_row(test_runs_csv, test_row)

        # Append to extended ablation CSV (with ckpt tag)
        append_to_ablation_csv_all_ckpts(cfg, test_res, ckpt_tag=tag)

        if tag == "BEST_HPACS":
            best_hpacs_test_res = test_res

    # Append exactly ONE row per ablation mode to the main Table 4.1 CSV.
    # Prefer BEST_HPACS if it exists (strict-val feasible). Otherwise, record BEST_SCORE with hpacs_ok=0.
    if best_hpacs_test_res is not None:
        append_to_ablation_csv(cfg, best_hpacs_test_res, ckpt_tag="BEST_HPACS", hpacs_ok=1, hpacs_reason="")
    else:
        # No strict-feasible checkpoint on VAL → baseline is infeasible under HPACS.
        # We still log BEST_SCORE in Table 4.1 with hpacs_ok=0 for completeness (still 5 rows).
        # Choose BEST_SCORE if available; else fall back to LAST.
        chosen_tag = None
        chosen_res = None
        for tag, _ in candidates:
            if tag == "BEST_SCORE":
                chosen_tag = tag
                break
        if chosen_tag == "BEST_SCORE":
            # BEST_SCORE was evaluated; re-load and re-evaluate once to be explicit and stable.
            load_trainable_state_dict(model, best_score_ckpt_path, device)
            chosen_res = evaluate_epoch(model, test_loader, device, cfg, output_viz_dir=None, report_dir=os.path.join(cfg.REPORT_DIR, "test_best_score_table"))
            append_to_ablation_csv(cfg, chosen_res, ckpt_tag="BEST_SCORE", hpacs_ok=0, hpacs_reason="NO_VAL_CKPT_PASSED_GATE")
            append_to_ablation_csv_all_ckpts(cfg, chosen_res, ckpt_tag="BEST_SCORE_TABLE")
            print("⚠️ No checkpoint satisfied strict HPACS gate on VAL; Table 4.1 uses BEST_SCORE with hpacs_ok=0.")
        else:
            load_trainable_state_dict(model, last_trainable_ckpt_path, device)
            chosen_res = evaluate_epoch(model, test_loader, device, cfg, output_viz_dir=None, report_dir=os.path.join(cfg.REPORT_DIR, "test_last_table"))
            append_to_ablation_csv(cfg, chosen_res, ckpt_tag="LAST", hpacs_ok=0, hpacs_reason="NO_VAL_CKPT_PASSED_GATE")
            append_to_ablation_csv_all_ckpts(cfg, chosen_res, ckpt_tag="LAST_TABLE")
            print("⚠️ No checkpoint satisfied strict HPACS gate on VAL; Table 4.1 uses LAST with hpacs_ok=0.")

    print("\n🎉 All processes completed successfully!")
    print(f"📁 CSV saved under: {cfg.REPORT_DIR}")
    print("  - train_val_history.csv   (每个 epoch 的 train_loss + Val 全指标)")
    print("  - best_events.csv         (每次 new best 的全指标快照)")
    print("  - best_summary.csv        (当前 best 的单行汇总)")
    print("  - test_runs.csv           (每次 final test 的全指标记录)")
    print("  - test_summary.csv        (当前 final test 的单行汇总)")


if __name__ == "__main__":
    main()