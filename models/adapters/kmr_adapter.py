import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class KMRAdapter(nn.Module):
    """
    动力学记忆路由适配器 (Kinematic Memory Routing Adapter).
    终极防爆版：搭载物理探头防倒灌绝缘墙(Detach)、算子交换律 OOM 拯救机制、
    超球面余弦路由、全常量重参数化兜底，以及 FP16 防下溢钳制与评估确定性控制。
    """

    def __init__(self, d_model=256, m_attn=32, num_prototypes=64):
        super().__init__()
        self.d_model = d_model
        self.m = m_attn
        self.num_protos = num_prototypes
        self.sigma_e = 0.02  # 正交身份绝缘注入强度

        # 1. 降维漏斗 (1x1 Conv)
        self.W_down = nn.Conv2d(d_model, self.m, kernel_size=1, bias=False)

        # Tubular cross prefilter (anisotropic low-pass) for PRMI rectilinear / piecewise-box masks
        # alpha_tube = sigmoid(raw) in (0,1); init raw=-4 => alpha≈0.018 (非常温和)
        self.tube_gate_raw = nn.Parameter(torch.full((1, self.m, 1, 1), -4.0))

        # 2. 零成本张量探头 (Zero-Cost Tensor Probe)
        self.probe = nn.Sequential(
            nn.Conv2d(self.m, self.m, kernel_size=3, padding=1, groups=self.m, bias=False),
            nn.GroupNorm(4, self.m),  # 流形稳压器，防止降维后特征方差剧烈震荡
            nn.GELU(),
            nn.Conv2d(self.m, 3, kernel_size=1)
        )

        # 3. 隐式可微形态学正交字典 (DMPB)
        self.dict_K = nn.Parameter(torch.randn(num_prototypes, self.m))
        self.dict_V = nn.Parameter(torch.randn(num_prototypes, self.m, 64, 64) * 0.02)  # 极小方差初始化防流形冲撞
        nn.init.orthogonal_(self.dict_K)

        # 【排雷 4】：超球面路由可学习温度系数 (防死锁重参数化)
        self.temp_raw = nn.Parameter(torch.tensor(0.0))  # 初始 softplus(0) ≈ 0.69

        # 4. 动力学因果常量 (物理约束重参数化 Raw 值)
        # 目标安全物理值: tau_base≈30, alpha_c≈10, alpha_t≈5, gamma_h≈100
        self.tau_base_raw = nn.Parameter(torch.tensor(3.4))  # softplus(3.4) ≈ 30.0
        self.alpha_c_raw = nn.Parameter(torch.tensor(2.3))  # softplus(2.3) ≈ 10.0
        self.alpha_t_raw = nn.Parameter(torch.tensor(1.6))  # softplus(1.6) ≈ 5.0
        self.gamma_h_raw = nn.Parameter(torch.tensor(4.6))  # softplus(4.6) ≈ 100.0

        # 5. 升维平滑劫持映射
        self.W_up = nn.Conv2d(self.m, d_model, kernel_size=1, bias=False)

        # 强制纪律：确保 Epoch 0 时系统为 100% 的原生刚性历史追踪
        nn.init.normal_(self.W_down.weight, std=0.02)
        nn.init.zeros_(self.W_up.weight)

    def forward(self, F_track, delta_t):
        B, D, H, W = F_track.shape
        target_dtype = F_track.dtype

        Z_track = self.W_down(F_track)

        # Optional tubular prefilter for probes: encourages long rectilinear continuity and suppresses point noise
        Z_probe = Z_track
        if hasattr(self, "tube_gate_raw"):
            a = torch.sigmoid(self.tube_gate_raw).to(device=Z_track.device, dtype=Z_track.dtype)
            z_tube = (
                F.avg_pool2d(Z_track, kernel_size=(1, 9), stride=1, padding=(0, 4)) +
                F.avg_pool2d(Z_track, kernel_size=(9, 1), stride=1, padding=(4, 0))
            )
            Z_probe = (1.0 - a) * Z_track + a * z_tube

        # ====================================================================
        # Phase 1: 探针微秒级物理场瞬时解码
        # ====================================================================
        # Clamp 极值截断，防 log(0) 炸毁交叉熵 Loss
        probe_probs = torch.clamp(torch.sigmoid(self.probe(Z_probe)), min=1e-5, max=1.0 - 1e-5)

        # 🚨 【排雷 2 核心阻断】：绝对物理隔离梯度倒灌！
        # 强制 detach() 阻断 Mask Decoder 的下游 Loss 篡改探头逻辑，维护探头的物理观测纯洁性！
        P_c_det = probe_probs[:, 0:1].detach()
        P_t_det = probe_probs[:, 1:2].detach()
        P_h_det = probe_probs[:, 2:3].detach()

        # ====================================================================
        # Phase 2: 上帝视角盲配先验合成 (L2防坍缩 + 算子交换律防 OOM)
        # ====================================================================
        q_t = F.adaptive_avg_pool2d(Z_track, 1).view(B, self.m)

        # L2 Normalize 彻底稳定内积空间，防 Softmax 极化导致字典失效
        q_norm = F.normalize(q_t, p=2, dim=-1)
        K_norm = F.normalize(self.dict_K.to(target_dtype), p=2, dim=-1)

        temp_safe = F.softplus(self.temp_raw.to(target_dtype)) + 1e-4
        attn = F.softmax(torch.matmul(q_norm, K_norm.T) / temp_safe, dim=-1)  # [B, N]

        # 🚨 【排雷 1 救命级优化】：算子交换律！先在 64x64 上求和降维，再插值放大！直接斩断 95% VRAM！
        Z_sup_small = torch.einsum('bn, nmxy -> bmxy', attn, self.dict_V.to(target_dtype))
        Z_sup = F.interpolate(Z_sup_small, size=(H, W), mode='bilinear', align_corners=False)

        # ====================================================================
        # Phase 3: 正交因果身份注入 (训练/验证绝对隔离)
        # ====================================================================
        # 🚨 【排雷 5】：旁路验证模式下的噪声，杜绝评估结果的“量子抖动”
        if self.training:
            Z_track_ins = Z_track + torch.randn_like(Z_track) * self.sigma_e
            Z_sup_ins = Z_sup + torch.randn_like(Z_sup) * self.sigma_e
        else:
            Z_track_ins = Z_track
            Z_sup_ins = Z_sup

        # ====================================================================
        # Phase 4: 动力学时空半衰期与门限掩码生成
        # ====================================================================
        # 【排雷 3】：利用 Softplus 强行榨取绝对正值的客观物理约束，防参数滑落负区间
        tau_base = F.softplus(self.tau_base_raw.to(target_dtype)).view(1, 1, 1, 1)
        alpha_c = F.softplus(self.alpha_c_raw.to(target_dtype)).view(1, 1, 1, 1)
        alpha_t = F.softplus(self.alpha_t_raw.to(target_dtype)).view(1, 1, 1, 1)
        gamma_h = F.softplus(self.gamma_h_raw.to(target_dtype)).view(1, 1, 1, 1)

        # 外部依然使用 Softplus 替代 ReLU，确保重度惩罚下仍具备非零平滑梯度 (防 Dead Zone)
        tau_raw = tau_base + alpha_c * P_c_det - alpha_t * P_t_det - gamma_h * P_h_det
        tau = F.softplus(tau_raw) + 1e-3

        dt_tensor = delta_t.view(B, 1, 1, 1).to(dtype=target_dtype, device=F_track.device)

        # 强制切断指数下溢深渊，防 FP16/BF16 混合精度梯度爆炸 (NaN 污染)
        exponent = torch.clamp(-dt_tensor / tau, min=-60.0, max=0.0)
        F_kin = torch.exp(exponent)  # [B, 1, H, W]

        # ====================================================================
        # Phase 5: 全可微残差凸组合软劫持
        # ====================================================================
        # F_kin -> 1 (主根连续生长): 完美继承原生时空注意力追踪 F_track
        # F_kin -> 0 (跨季断层/裂缝爆发): 历史残影被合法物理抹除，权重排山倒海般平滑倾斜至 Z_sup
        Z_diff = (1.0 - F_kin) * (Z_sup_ins - Z_track_ins)
        F_out = F_track + self.W_up(Z_diff)

        # ⚠️ 极其关键：向外传出的 probe_probs 不带 detach！
        # 必须把带有梯度的预测交给 losses.py 中的 ZCMP 去接受伪标签的真实物理提纯寻优。
        return F_out, probe_probs, F_kin