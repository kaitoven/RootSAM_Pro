import cv2
import torch
import numpy as np
import torch.nn.functional as F


class PhysicalPreservingTransforms:
    def __init__(self, target_size=1024):
        self.target_size = target_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def apply_image(self, image_np):
        h, w = image_np.shape[:2]
        scale = self.target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_h, pad_w = self.target_size - new_h, self.target_size - new_w
        top, bottom = pad_h // 2, pad_h - (pad_h // 2)
        left, right = pad_w // 2, pad_w - (pad_w // 2)
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        padded_norm = (padded.astype(np.float32) / 255.0 - self.mean) / self.std
        tensor = torch.from_numpy(padded_norm).permute(2, 0, 1).float()

        pad_info = torch.tensor([top, bottom, left, right, h, w], dtype=torch.long)
        return tensor, pad_info

    @staticmethod
    def reverse_logits_to_physical(logits_tensor, pad_info_tensor):
        """返回张量形状严格保证为: [1, C, H_orig, W_orig]"""
        top, bottom, left, right, orig_h, orig_w = pad_info_tensor.tolist()
        h, w = logits_tensor.shape[-2:]

        # 🚨 【排雷 5】：极其关键的切片护盾！利用绝对坐标边界计算，彻底防止 python list[:-0] 导致的特征空域坍缩！
        y1, y2 = top, h - bottom if bottom > 0 else h
        x1, x2 = left, w - right if right > 0 else w

        unpadded = logits_tensor[..., y1:y2, x1:x2]
        recovered = F.interpolate(unpadded, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        return recovered