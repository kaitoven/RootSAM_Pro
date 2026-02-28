import torch
import torch.nn.functional as F


# ----------------------------
# Robust FPN layout inference
# ----------------------------
_CSET = (8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 512, 640, 768, 1024)

def _layout_score(values, assume: str) -> float:
    # Lower is better. assume in {'bchw','bhwc'}.
    items = []
    for v in values:
        if not (torch.is_tensor(v) and v.dim() == 4):
            continue
        if assume == 'bchw':
            c = int(v.shape[1]); area = int(v.shape[2]) * int(v.shape[3])
        else:
            c = int(v.shape[3]); area = int(v.shape[1]) * int(v.shape[2])
        items.append((area, c))
    if len(items) <= 1:
        return 0.0
    items.sort(key=lambda x: x[0], reverse=True)
    violations = 0
    invalid_c = 0
    prev_c = items[0][1]
    if prev_c <= 0 or prev_c > 2048:
        invalid_c += 1
    for _, c in items[1:]:
        if c < prev_c:
            violations += 1
        if c <= 0 or c > 2048:
            invalid_c += 1
        prev_c = c
    typical_miss = sum(1 for _, c in items if c not in _CSET)
    return float(violations) + 0.25 * float(invalid_c) + 0.05 * float(typical_miss)

def _infer_fpn_layout(values) -> str:
    s_bchw = _layout_score(values, 'bchw')
    s_bhwc = _layout_score(values, 'bhwc')
    return 'bhwc' if s_bhwc < s_bchw else 'bchw'


def _get_mask_decoder(sam2):
    md = getattr(sam2, "mask_decoder", getattr(sam2, "sam_mask_decoder", None))
    if md is None:
        raise AttributeError("SAM2 has no mask decoder.")
    return md


def _collect_fpn_feats_bchw(high_res_features):
    feats = []
    if isinstance(high_res_features, dict):
        values = list(high_res_features.values())
    elif isinstance(high_res_features, (list, tuple)):
        values = list(high_res_features)
    else:
        values = []

    layout = _infer_fpn_layout([v for v in values if torch.is_tensor(v) and v.dim() == 4])
    for v in values:
        if not (torch.is_tensor(v) and v.dim() == 4):
            continue
        if layout == 'bhwc':
            v = v.permute(0, 3, 1, 2).contiguous()
        feats.append(v)
    return feats


def _pick_or_interp(feats_all, target_hw, prefer_c=None):
    th, tw = int(target_hw[0]), int(target_hw[1])
    best, best_score = None, None
    for f in feats_all:
        if prefer_c is not None and int(f.shape[1]) != int(prefer_c):
            continue
        h, w = int(f.shape[-2]), int(f.shape[-1])
        score = abs(h - th) + abs(w - tw)
        if best is None or score < best_score:
            best, best_score = f, score
    if best is None:
        return None
    if best.shape[-2:] != (th, tw):
        best = F.interpolate(best, size=(th, tw), mode="bilinear", align_corners=False)
    return best


def decode_masks_compat(
    *,
    sam2,
    image_embeddings,
    image_pe,
    sparse_prompt_embeddings,
    dense_prompt_embeddings,
    multimask_output,
    high_res_features,
    # optional: washer gates (nn.Parameter on host); if None -> disabled
    srd_washer_g0_raw=None,
    srd_washer_g1_raw=None,
    # optional: BHFI module (callable) to refine decoder logits
    bhfi=None,
):
    """Compat shim across SAM2.1 versions + robust HR feature selection + optional washer + optional BHFI.

    Keeps RootSAMPro lean: this file owns signature juggling, HR feature selection, and post-decoder refinement.
    """
    mask_decoder = _get_mask_decoder(sam2)

    if torch.is_tensor(image_pe) and image_pe.dim() >= 1 and image_pe.size(0) != 1:
        image_pe = image_pe[:1]

    feats_all = _collect_fpn_feats_bchw(high_res_features)
    H, W = int(image_embeddings.shape[-2]), int(image_embeddings.shape[-1])
    s1_hw, s0_hw = (H * 2, W * 2), (H * 4, W * 4)

    # Shield: prefer exact channels (only pass HR features when we recover correct shapes)
    feat_s1 = _pick_or_interp(feats_all, s1_hw, prefer_c=64) if len(feats_all) > 0 else None
    feat_s0 = _pick_or_interp(feats_all, s0_hw, prefer_c=32) if len(feats_all) > 0 else None

    use_hr = (feat_s0 is not None) and (feat_s1 is not None)
    if not use_hr:
        feat_s0, feat_s1 = None, None

    # ------------------------------------------------------------
    # HR-FPN Soil Washers (learnable low-pass convex blend)
    # feat <- (1-alpha)*feat + alpha*avgpool(feat)
    # ------------------------------------------------------------
    if use_hr and (srd_washer_g1_raw is not None) and (feat_s1 is not None):
        a1 = torch.sigmoid(srd_washer_g1_raw).to(device=feat_s1.device, dtype=feat_s1.dtype)
        feat_s1 = (1.0 - a1) * feat_s1 + a1 * F.avg_pool2d(feat_s1, kernel_size=3, stride=1, padding=1)
    if use_hr and (srd_washer_g0_raw is not None) and (feat_s0 is not None):
        a0 = torch.sigmoid(srd_washer_g0_raw).to(device=feat_s0.device, dtype=feat_s0.dtype)
        feat_s0 = (1.0 - a0) * feat_s0 + a0 * F.avg_pool2d(feat_s0, kernel_size=3, stride=1, padding=1)

    hr = (feat_s0, feat_s1) if use_hr else None

    def _refine(dec_out):
        if bhfi is None:
            return dec_out
        try:
            return bhfi(dec_out, feat_s0, feat_s1) if use_hr else bhfi(dec_out, None, None)
        except Exception:
            return dec_out

    try:
        if use_hr:
            dec_out = mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_prompt_embeddings,
                dense_prompt_embeddings=dense_prompt_embeddings,
                multimask_output=multimask_output,
                high_res_features=hr,
            )
        else:
            dec_out = mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_prompt_embeddings,
                dense_prompt_embeddings=dense_prompt_embeddings,
                multimask_output=multimask_output,
            )
        return _refine(dec_out)
    except TypeError as e:
        # older signatures
        if "repeat_image" in str(e):
            try:
                if use_hr:
                    dec_out = mask_decoder(
                        image_embeddings=image_embeddings,
                        image_pe=image_pe,
                        sparse_prompt_embeddings=sparse_prompt_embeddings,
                        dense_prompt_embeddings=dense_prompt_embeddings,
                        multimask_output=multimask_output,
                        repeat_image=False,
                        high_res_features=hr,
                    )
                else:
                    dec_out = mask_decoder(
                        image_embeddings=image_embeddings,
                        image_pe=image_pe,
                        sparse_prompt_embeddings=sparse_prompt_embeddings,
                        dense_prompt_embeddings=dense_prompt_embeddings,
                        multimask_output=multimask_output,
                        repeat_image=False,
                    )
                return _refine(dec_out)
            except TypeError:
                dec_out = mask_decoder(image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, multimask_output, False, hr)
                return _refine(dec_out)
        try:
            dec_out = mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_prompt_embeddings,
                dense_prompt_embeddings=dense_prompt_embeddings,
                multimask_output=multimask_output,
            )
            return _refine(dec_out)
        except Exception:
            dec_out = mask_decoder(image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, multimask_output)
            return _refine(dec_out)
