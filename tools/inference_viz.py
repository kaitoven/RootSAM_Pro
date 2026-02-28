import numpy as np
import matplotlib

# 🚨 【排雷 5】：强制声明无头渲染后端，防止 Linux 集群报 _tkinter.TclError 崩溃
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render_beyond_gt_proof(raw_img_rgb, mask_gt, mask_pred, save_path):
    """定性验证：红色为人类错漏真值，绿色为 RootSAM-Pro 越级预测的丝滑拓扑。"""

    # 强制转换为 float32 防止 uint8 相乘溢出与色彩断层
    overlay = raw_img_rgb.copy().astype(np.float32)

    gt_only = np.logical_and(mask_gt == 1, mask_pred == 0)
    pred_only = np.logical_and(mask_pred == 1, mask_gt == 0)
    tp = np.logical_and(mask_pred == 1, mask_gt == 1)

    overlay[gt_only] = overlay[gt_only] * 0.4 + np.array([255.0, 0.0, 0.0]) * 0.6
    overlay[pred_only] = overlay[pred_only] * 0.4 + np.array([0.0, 255.0, 0.0]) * 0.6
    overlay[tp] = overlay[tp] * 0.4 + np.array([255.0, 255.0, 0.0]) * 0.6

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(raw_img_rgb);
    axes[0].set_title("Raw Minirhizotron Image");
    axes[0].axis('off')

    gt_vis = raw_img_rgb.copy().astype(np.float32)
    gt_vis[mask_gt == 1] = gt_vis[mask_gt == 1] * 0.5 + np.array([255.0, 0.0, 0.0]) * 0.5
    gt_vis = np.clip(gt_vis, 0, 255).astype(np.uint8)

    axes[1].imshow(gt_vis);
    axes[1].set_title("Flawed WinRHIZO GT (Red)");
    axes[1].axis('off')
    axes[2].imshow(overlay);
    axes[2].set_title("RootSAM-Pro Beyond-GT (Green)");
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # 防止内存泄漏