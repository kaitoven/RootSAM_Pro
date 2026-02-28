import os
import cv2
import json
import torch
import random
import math
import numpy as np
from datetime import datetime
from collections import defaultdict
from torch.utils.data import Dataset, Sampler
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from utils.transforms import PhysicalPreservingTransforms


class PRMI_KinematicDataset(Dataset):
    def __init__(self, root_dir, subset_name, split='train', seq_length=4, target_size=1024, train_mode: str = 'CLIP', train_max_seq_len: int = 0):
        # 1. 绝对的官方路径隔离
        self.split = split
        self.img_dir = os.path.join(root_dir, split, 'images', subset_name)
        self.mask_dir = os.path.join(root_dir, split, 'masks_pixel_gt', subset_name)
        self.json_path = os.path.join(root_dir, split, 'labels_image_gt', f"{subset_name}_{split}.json")

        self.seq_length = seq_length
        self.subset_name = subset_name
        self.transform = PhysicalPreservingTransforms(target_size)
        self.disk_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

        self.train_mode = str(train_mode).upper()
        self.train_max_seq_len = int(train_max_seq_len) if train_max_seq_len is not None else 0
        # 2. 严格在当前 split 内部重组物理因果线
        self.tubes = self._build_causal_sequences()
        self.seq_stats = self._build_seq_stats()

        # 🚨 【防泄漏与防重复计算核心逻辑】
        if self.split == 'train':
            # TRAIN_MODE controls whether we train on fixed-length clips or full sequences.
            if self.train_mode == 'SEQUENCE':
                self.snippets = self._generate_train_sequences(max_len=self.train_max_seq_len)
            else:
                self.snippets = self._generate_sliding_snippets()  # 训练期：定长滑动切片用于 Batch 并行
        else:
            self.snippets = self._generate_eval_sequences()  # Val/Test：原链无损吐出防重复计数

    def _build_causal_sequences(self):
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        tubes = defaultdict(list)
        # for item in data:
        #     seq_id = f"{item['crop']}_{item['location']}_{item['tube_num']}_{item['depth']}"
        #     item['timestamp'] = datetime.strptime(f"{item['date']}_{item['image_name'].split('_')[4]}",
        #                                           "%Y.%m.%d_%H%M%S")
        #     tubes[seq_id].append(item)
        # ✅ Strict causal key: keep location to prevent cross-site tube collisions
        for item in data:
            crop = item.get('crop', '')
            loc = item.get('location', '')
            tube = item.get('tube_num', '')
            depth = item.get('depth', '')
            seq_id = f"{crop}_{loc}_{tube}_{depth}"
            item['timestamp'] = datetime.strptime(
                f"{item['date']}_{item['image_name'].split('_')[4]}",
                "%Y.%m.%d_%H%M%S",
            )
            tubes[seq_id].append(item)
        for seq_id, frames in tubes.items():
            frames.sort(key=lambda x: x['timestamp'])
            for i in range(len(frames)):
                # 完全对齐论文要求：该 split 内的首帧强制 NA (999.0 物理截断)
                frames[i]['delta_t'] = 999.0 if i == 0 else (frames[i]['timestamp'] - frames[i - 1][
                    'timestamp']).total_seconds() / 86400.0
        return tubes

    def _build_seq_stats(self):
        """预计算每条 tube-depth 序列的长度、flips(mixed) 等分层统计字段。

        对齐《PRMI（去除 Switchgrass-300）分层评估》：
          - seq_len：序列帧数
          - flips：序列内 has_root 的 0/1 变化次数
          - mixed：flips>=1
        """
        stats = {}
        for seq_id, frames in self.tubes.items():
            has = [int(f.get('has_root', 0)) for f in frames]
            flips = 0
            for i in range(1, len(has)):
                flips += int(has[i] != has[i - 1])
            stats[seq_id] = {
                'seq_len': len(frames),
                'flips': flips,
                'mixed': flips >= 1,
                'has_root_any': any(h == 1 for h in has),
            }
        return stats

    def _generate_sliding_snippets(self):
        """仅用于 Train 阶段的定长截断与特征扩增引擎"""
        snippets = []
        for seq_id, frames in self.tubes.items():
            if len(frames) == 1:
                snippets.append({'type': 'insular', 'seq_id': seq_id, 'start_idx': 0, 'frames': [frames[0]] * self.seq_length})
            else:
                for i in range(max(1, len(frames) - self.seq_length + 1)):
                    clip = frames[i: i + self.seq_length]
                    while len(clip) < self.seq_length: clip.append(clip[-1])
                    max_dt = max([f['delta_t'] for f in clip if f['delta_t'] != 999.0] + [0])

                    # ✅ Step-A：物理时序状态严密分层
                    frame_has_roots = [int(f.get('has_root', 0)) for f in clip]

                    # 1) 严苛 pure-soil：必须 100% 全片无根
                    is_pure_soil = all(r == 0 for r in frame_has_roots)

                    # 2) dynamic：片段内 0↔1 翻转（长出/消亡瞬间），高价值困难样本
                    is_dynamic = (any(r == 0 for r in frame_has_roots) and any(r == 1 for r in frame_has_roots))

                    if is_pure_soil:
                        sType = 'pure_soil'
                    elif is_dynamic:
                        sType = 'dynamic'
                    else:
                        sType = 'gap' if max_dt > 90 else 'tracking'

                    snippets.append({'type': sType, 'seq_id': seq_id, 'start_idx': i, 'frames': clip})
        return snippets

    def _generate_train_sequences(self, max_len: int = 0):
        """Training-time sequence generator.

        - If max_len<=0: return each full causal sequence as one sample (rollout training).
        - If max_len>0 and a sequence is longer than max_len: generate causal windows of length max_len
          (still within the same seq_id; no cross-seq mixing).
        """
        snippets = []
        max_len = int(max_len) if max_len is not None else 0
        for seq_id, frames in self.tubes.items():
            st = self.seq_stats.get(seq_id, {'seq_len': len(frames), 'flips': 0, 'mixed': False, 'has_root_any': False})
            # sequence-level type (for optional curriculum/sampling)
            if not bool(st.get('has_root_any', False)):
                typ = 'pure_soil'
            elif bool(st.get('mixed', False)) or int(st.get('flips', 0)) >= 1:
                typ = 'dynamic'
            else:
                typ = 'tracking'

            if max_len <= 0 or len(frames) <= max_len:
                snippets.append({'type': typ, 'seq_id': seq_id, 'start_idx': 0, 'frames': frames})
            else:
                stride = max(1, max_len // 2)
                L = len(frames)
                for s in range(0, max(1, L - max_len + 1), stride):
                    window = frames[s:s + max_len]
                    if len(window) < max_len:
                        break
                    snippets.append({'type': typ, 'seq_id': seq_id, 'start_idx': int(s), 'frames': window})
                # ensure tail coverage
                if (L - max_len) > 0:
                    tail = frames[-max_len:]
                    snippets.append({'type': typ, 'seq_id': seq_id, 'start_idx': int(L - max_len), 'frames': tail})
        return snippets

    def _generate_eval_sequences(self):
        """🚨 仅用于 Val/Test 阶段的严密评估引擎，绝对防止一帧被评估多次！"""
        snippets = []
        for seq_id, frames in self.tubes.items():
            # 直接将整条真实因果链作为一个独立样本，长度 T 是动态的 (例如 1 帧、3 帧或 13 帧)
            snippets.append({'type': 'eval_seq', 'seq_id': seq_id, 'start_idx': 0, 'frames': frames})
        return snippets

    def _on_the_fly_physics(self, image, mask_path, has_root):
        H, W = image.shape[:2]
        if has_root == 0:
            m = np.zeros((H, W), dtype=np.float32)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edge = (cv2.Canny(gray, 50, 150) > 0).astype(np.float32)
            return m, m, m, edge, np.ones_like(m)

        m = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
        P_c = skeletonize(m).astype(np.float32)
        P_t = cv2.dilate((cv2.cornerHarris(m, 2, 3, 0.04) > 0).astype(np.float32), np.ones((3, 3)))
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        P_h = np.clip((cv2.Canny(gray, 50, 150) > 0).astype(np.float32) - cv2.dilate(m, self.disk_kernel), 0, 1)

        # 👑【方案A 对齐】：SDF-Guided Exemption Field（边界松弛豁免场）
        # 约定：W_SDF ∈ [0,1]，越接近 0 表示越豁免（边界/毛刺区），越接近 1 表示越严格（远离边界的体积区）
        # 使用近似 Signed Distance：sdf = d_out - d_in，边界处 |sdf|≈0。
        d_out = distance_transform_edt(1.0 - m)  # outside distance to nearest root
        d_in = distance_transform_edt(m)         # inside distance to nearest background
        abs_sdf = np.abs(d_out - d_in)
        sigma = 5.0
        W_SDF = 1.0 - np.exp(-(abs_sdf ** 2) / (2.0 * sigma ** 2))
        W_SDF = np.clip(W_SDF, 0.0, 1.0).astype(np.float32)
        return m, P_c, P_t, P_h, W_SDF

    def __len__(self):
        return len(self.snippets)

    def __getitem__(self, idx):
        snip = self.snippets[idx]
        clip = snip['frames']
        seq_id = snip.get('seq_id', 'NA')
        start_idx = int(snip.get('start_idx', 0))
        st = self.seq_stats.get(seq_id, {'seq_len': len(clip), 'flips': 0, 'mixed': False, 'has_root_any': False})
        t_img, t_mask, t_dt, t_pc, t_pt, t_ph, t_wsdf, t_pad = [], [], [], [], [], [], [], []
        t_meta = []

        for j, frame in enumerate(clip):
            image = cv2.cvtColor(cv2.imread(os.path.join(self.img_dir, frame['image_name'])), cv2.COLOR_BGR2RGB)
            m, pc, pt, ph, wsdf = self._on_the_fly_physics(image,
                                                           os.path.join(self.mask_dir, frame['binary_mask']) if frame[
                                                               'has_root'] else None, frame['has_root'])

            img_tensor, pad_info = self.transform.apply_image(image)
            t_img.append(img_tensor)
            t_pad.append(pad_info)
            t_dt.append(frame['delta_t'])
            t_mask.append(torch.from_numpy(m).unsqueeze(0).float())
            t_pc.append(torch.from_numpy(pc).unsqueeze(0).float())
            t_pt.append(torch.from_numpy(pt).unsqueeze(0).float())
            t_ph.append(torch.from_numpy(ph).unsqueeze(0).float())
            t_wsdf.append(torch.from_numpy(wsdf).unsqueeze(0).float())

            # ===== 顶刊级可复现元信息（用于分层评估/TEPR） =====
            ts = frame.get('timestamp', None)
            ts_iso = ts.isoformat() if hasattr(ts, 'isoformat') else (str(ts) if ts is not None else "")
            t_meta.append({
                'subset': self.subset_name,
                'split': self.split,
                'seq_id': seq_id,
                'seq_len': int(st['seq_len']),
                'flips': int(st['flips']),
                'mixed': bool(st['mixed']),
                'frame_idx': int(start_idx + j),
                'delta_t': float(frame.get('delta_t', 999.0)),
                'has_root': int(frame.get('has_root', 0)),
                'crop': frame.get('crop', ''),
                'location': frame.get('location', ''),
                'tube_num': frame.get('tube_num', ''),
                'depth': frame.get('depth', ''),
                'date': frame.get('date', ''),
                'timestamp': ts_iso,
                'image_name': frame.get('image_name', ''),
            })

        return {'images': torch.stack(t_img), 'masks_gt': torch.stack(t_mask), 'delta_t': torch.tensor(t_dt).float(),
                'P_c': torch.stack(t_pc), 'P_t': torch.stack(t_pt), 'P_h': torch.stack(t_ph),
                'W_SDF': torch.stack(t_wsdf), 'pad_info': torch.stack(t_pad),
                'raw_paths': [os.path.join(self.img_dir, f['image_name']) for f in clip],
                'meta': t_meta}


# (ExtremeCurriculumSampler 保持之前代码不变，仅在训练时使用)
class ExtremeCurriculumSampler(Sampler):
    def __init__(self, ds, bs=None, batch_size=None):
        # 兼容两种调用方式：ExtremeCurriculumSampler(ds, bs) 或 ExtremeCurriculumSampler(ds, batch_size=...)
        if bs is None:
            bs = batch_size
        assert bs is not None, "ExtremeCurriculumSampler requires bs or batch_size"
        self.ds, self.bs = ds, bs
        self.i_trk = [i for i, s in enumerate(ds.snippets) if s['type'] == 'tracking']
        self.i_ins = [i for i, s in enumerate(ds.snippets) if s['type'] in ['insular', 'gap', 'dynamic']]
        self.i_sol = [i for i, s in enumerate(ds.snippets) if s['type'] == 'pure_soil']

    def __iter__(self):
        n_tr, n_in = int(self.bs * 0.5), int(self.bs * 0.3)
        n_so = self.bs - n_tr - n_in
        p_trk, p_ins, p_sol = self.i_trk.copy(), self.i_ins.copy(), self.i_sol.copy()
        random.shuffle(p_trk);
        random.shuffle(p_ins);
        random.shuffle(p_sol)

        for _ in range(len(self.ds) // self.bs):
            b = []
            b += random.choices(p_trk, k=n_tr) if p_trk else random.choices(p_ins, k=n_tr)
            b += random.choices(p_ins, k=n_in) if p_ins else random.choices(p_trk, k=n_in)
            b += random.choices(p_sol, k=n_so) if p_sol else random.choices(p_trk, k=n_so)
            random.shuffle(b);
            yield b

    def __len__(self): return len(self.ds) // self.bs

class GroupBySeqLenBatchSampler(Sampler):
    """BatchSampler that groups samples by exact sequence length.

    Purpose:
      - val/test often return variable-length full sequences (eval_seq)
      - we must NOT pad across different seq_len, otherwise frame indices misalign
      - this sampler yields batches where all items share the same T

    Notes:
      - does NOT mix frames across sequences; each dataset item is one seq/clip
      - safe for batch_size>1 without touching trainer loops
    """

    def __init__(self, ds: PRMI_KinematicDataset, batch_size: int = 1, shuffle: bool = False, drop_last: bool = False):
        assert batch_size >= 1
        self.ds = ds
        self.bs = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)

        # bucket indices by exact length T
        buckets = defaultdict(list)
        for i, snip in enumerate(getattr(ds, "snippets", [])):
            try:
                T = len(snip.get("frames", []))
            except Exception:
                T = getattr(ds, "seq_length", 1)
            if T <= 0:
                T = 1
            buckets[int(T)].append(i)
        self.buckets = buckets
        self._len = 0
        for T, idxs in self.buckets.items():
            n = len(idxs)
            self._len += (n // self.bs) if (self.drop_last) else int(math.ceil(n / self.bs))

    def __iter__(self):
        # local copy to avoid in-place edits
        all_T = list(self.buckets.keys())
        if self.shuffle:
            random.shuffle(all_T)

        for T in all_T:
            idxs = list(self.buckets[T])
            if self.shuffle:
                random.shuffle(idxs)
            # chunk
            for s in range(0, len(idxs), self.bs):
                batch = idxs[s:s + self.bs]
                if self.drop_last and len(batch) < self.bs:
                    continue
                yield batch

    def __len__(self):
        return int(self._len)