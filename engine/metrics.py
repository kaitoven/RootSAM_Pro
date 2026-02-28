import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import binary_dilation


def _safe_div(a, b, eps=1e-8):
    return float(a) / float(b + eps)


def compute_frame_metrics(pred_mask_bool: np.ndarray, gt_mask_bool: np.ndarray, relaxation_delta: int = 5):
    """逐帧指标计算（用于输出 per-frame CSV 与分层评估）。

    返回字段尽量与顶刊常用报表一致：IoU/Dice/Precision/Recall + clDice + SDF-relaxed IoU。
    注：SDF-relaxed IoU 在这里用形态学膨胀近似 tolerance band（与现有 tracker 保持一致）。
    """
    pred_bool = pred_mask_bool.astype(bool)
    gt_bool = gt_mask_bool.astype(bool)

    sum_pred = int(pred_bool.sum())
    sum_gt = int(gt_bool.sum())

    inter = int(np.logical_and(pred_bool, gt_bool).sum())
    union = int(np.logical_or(pred_bool, gt_bool).sum())

    tp = inter
    fp = sum_pred - inter
    fn = sum_gt - inter
    tn = int(np.logical_and(~pred_bool, ~gt_bool).sum())

    iou = 1.0 if (union == 0 and sum_pred == 0 and sum_gt == 0) else (_safe_div(inter, union) if union > 0 else 0.0)
    dice = _safe_div(2 * inter, (sum_pred + sum_gt)) if (sum_pred + sum_gt) > 0 else (1.0 if union == 0 else 0.0)
    precision = _safe_div(tp, (tp + fp)) if (tp + fp) > 0 else (1.0 if sum_pred == 0 and sum_gt == 0 else 0.0)
    recall = _safe_div(tp, (tp + fn)) if (tp + fn) > 0 else (1.0 if sum_pred == 0 and sum_gt == 0 else 0.0)

    # relaxed IoU
    if sum_gt == 0:
        r_iou = 1.0 if sum_pred == 0 else 0.0
    else:
        tol_band = binary_dilation(gt_bool, iterations=int(relaxation_delta))
        r_inter = int(np.logical_and(pred_bool, tol_band).sum())
        r_iou = _safe_div(r_inter, union) if union > 0 else 0.0

    # clDice
    if sum_gt == 0 and sum_pred == 0:
        cldice = 1.0
    elif sum_gt == 0 or sum_pred == 0:
        cldice = 0.0
    else:
        t_pred = skeletonize(pred_bool).astype(np.float32)
        t_gt = skeletonize(gt_bool).astype(np.float32)
        t_prec = float(np.sum(t_pred * gt_bool)) / (float(np.sum(t_pred)) + 1e-8)
        t_sens = float(np.sum(t_gt * pred_bool)) / (float(np.sum(t_gt)) + 1e-8)
        cldice = float(2.0 * t_prec * t_sens / (t_prec + t_sens + 1e-8))

    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'pred_pixels': sum_pred, 'gt_pixels': sum_gt,
        'iou': float(iou), 'dice': float(dice), 'precision': float(precision), 'recall': float(recall),
        'sdf_relaxed_iou': float(r_iou),
        'cldice': float(cldice),
    }


class BeyondGT_MetricsTracker:
    """
    专为微根管长尾物理绝境量身定制的 5D 统计学追踪器。
    完美兼容 传统基础视觉指标 (Acc, P, R, F1, IoU) 与 顶刊特异性越级指标 (clDice, SDF-IoU)。
    采用全局像素级累积 (Global Accumulation)，彻底消灭极端不平衡下的除零崩溃 (NaN)。
    """

    def __init__(self, relaxation_delta=5):
        self.delta = relaxation_delta
        self.reset()

    def reset(self):
        # 1. 传统基线度量衡 (全局像素累积)
        self.tp = 0.0  # True Positives
        self.fp = 0.0  # False Positives
        self.fn = 0.0  # False Negatives
        self.tn = 0.0  # True Negatives

        # 2. 超越真值与微观拓扑度量
        self.relaxed_ious = []
        self.cldices = []

        # 3. 长尾绝境特异性靶向度量
        self.pure_soil_fp_pixels = 0
        self.pure_soil_total_pixels = 0
        self.insular_gap_recalls = []
        self.gap_ious = []

    def update(self, pred_mask, gt_mask, delta_t):
        """逐样本更新物理统计器"""
        pred_bool = pred_mask > 0.5
        gt_bool = gt_mask > 0.5

        sum_pred = np.sum(pred_bool)
        sum_gt = np.sum(gt_bool)

        # ==========================================
        # 模块 1：传统基础混淆矩阵 (全局像素累积)
        # ==========================================
        inter = np.logical_and(pred_bool, gt_bool).sum()
        union = np.logical_or(pred_bool, gt_bool).sum()

        self.tp += inter
        self.fp += (sum_pred - inter)
        self.fn += (sum_gt - inter)
        self.tn += np.logical_and(~pred_bool, ~gt_bool).sum()

        # 单图标准 IoU (供特定绝境统计使用)
        single_iou = inter / union if union > 0 else (1.0 if sum_gt == 0 and sum_pred == 0 else 0.0)

        # ==========================================
        # 模块 2：越级边界松弛保真度 (SDF-Relaxed IoU)
        # ==========================================
        if sum_gt == 0:
            r_iou = 1.0 if sum_pred == 0 else 0.0
        else:
            # 膨胀粗糙真值得到合法的物理松弛豁免带
            tol_band = binary_dilation(gt_bool, iterations=self.delta)
            r_inter = np.logical_and(pred_bool, tol_band).sum()
            r_iou = r_inter / union
        self.relaxed_ious.append(r_iou)

        # ==========================================
        # 模块 3：微观流体力学拓扑连通度 (clDice)
        # ==========================================
        if sum_gt == 0 and sum_pred == 0:
            self.cldices.append(1.0)
        elif sum_gt == 0 or sum_pred == 0:
            self.cldices.append(0.0)
        else:
            t_pred = skeletonize(pred_bool).astype(np.float32)
            t_gt = skeletonize(gt_bool).astype(np.float32)
            t_prec = np.sum(t_pred * gt_bool) / (np.sum(t_pred) + 1e-8)
            t_sens = np.sum(t_gt * pred_bool) / (np.sum(t_gt) + 1e-8)
            self.cldices.append(2.0 * t_prec * t_sens / (t_prec + t_sens + 1e-8))

        # ==========================================
        # 模块 4：靶向分流 - 长尾绝境特异性度量
        # ==========================================
        # 绝境 A: 纯土期防伪影能力 (SRD 模块核心证明)
        if sum_gt == 0:
            self.pure_soil_fp_pixels += sum_pred
            self.pure_soil_total_pixels += pred_bool.size

        # 绝境 B: 单帧孤岛(999.0) 或 跨季大断层(>90.0) (KMR 模块核心证明)
        if sum_gt > 0 and (delta_t == 999.0 or delta_t > 90.0):
            self.insular_gap_recalls.append(inter / sum_gt)
        if sum_gt > 0 and (90.0 < delta_t < 999.0):
            self.gap_ious.append(single_iou)

    # ✅ 关键修复：兼容 trainer.py 调用 summarize()
    def summarize(self):
        """Compatibility alias for older trainer code."""
        return self.compute_summary()

    def compute_summary(self):
        """一键生成大满贯报表 (直接对应论文 Table 结果)"""
        eps = 1e-8

        # 全局基础指标计算 (绝对防 NaN)
        accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn + eps)
        precision = self.tp / (self.tp + self.fp + eps) if (self.tp + self.fp) > 0 else 0.0
        recall = self.tp / (self.tp + self.fn + eps) if (self.tp + self.fn) > 0 else 0.0
        f1_score = 2.0 * self.tp / (2.0 * self.tp + self.fp + self.fn + eps)
        global_iou = self.tp / (self.tp + self.fp + self.fn + eps)

        fpr_soil = (
            self.pure_soil_fp_pixels / self.pure_soil_total_pixels
        ) * 100 if self.pure_soil_total_pixels > 0 else 0.0

        return {
            # --- 传统基准护城河 (Traditional Baselines) ---
            "Accuracy": accuracy * 100,
            "Precision": precision * 100,
            "Recall": recall * 100,
            "F1_Score": f1_score * 100,
            "Standard_IoU": global_iou * 100,

            # --- 顶刊前沿破局区 (Beyond-GT Innovations) ---
            "SDF_Relaxed_IoU": np.mean(self.relaxed_ious) * 100 if self.relaxed_ious else 0.0,
            "clDice": np.mean(self.cldices) * 100 if self.cldices else 0.0,

            # --- 物理绝境特异性指标 (Extreme Long-tail Specificity) ---
            "Pure_Soil_FPR": fpr_soil,  # 该指标越低越好
            "Insular_Gap_Recall": np.mean(self.insular_gap_recalls) * 100 if self.insular_gap_recalls else 0.0,
            "Gap_mIoU": np.mean(self.gap_ious) * 100 if self.gap_ious else 0.0
        }