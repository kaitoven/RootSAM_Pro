import torch
import torch.nn.functional as F


class ReflexMechanisms:
    @staticmethod
    def guided_fusion_unit(M_raw_logits, P_c_prob, kappa):
        """
        拓扑引力融合单元 (Topological Gravity Fusion).
        【排雷 2】：彻底摒弃 Sigmoid-Log 坍缩反演，直接在 Logit 负半轴劫持！
        """
        # 强制切断梯度倒灌，保全 ZCMP 探头的微观物理先验纯洁性
        S_pred = P_c_prob.detach().clamp(0.0, 1.0)

        # F.softplus(-M_raw_logits): 只有当大模型预测为背景(Logit<0)时，才产生正向修补拉力。
        # 若大模型已经预测为前景(Logit>0)，Softplus趋近于0，完美保护 SA-1B 原生平滑边界流形！
        gravity_force = F.softplus(-M_raw_logits)

        # 优美的偏微分残差注入，绝对不会引发 NaN 溢出
        M_fused_logits = M_raw_logits + kappa * S_pred * gravity_force

        return M_fused_logits

    @staticmethod
    def causal_firewall(P_c_prob, eta_energy=50.0, theta_frag=0.8):
        """
        零参数因果防火墙 (Zero-Param Causal Firewall).
        【排雷 4】：利用张量池化差分，在 GPU 内微秒级推演拓扑碎片化指数 C_frag。
        """
        S_pred = P_c_prob.detach().clamp(0.0, 1.0)

        # 1. 宏观拓扑能量积分 (Area)
        E_root = S_pred.sum(dim=(1, 2, 3))

        # 2. 边缘周长提取 (Perimeter)
        # 膨胀一次后减去原图，剩下的就是单像素边缘
        dilated_S = F.max_pool2d(S_pred, kernel_size=3, stride=1, padding=1)
        perimeter_edges = dilated_S - S_pred
        E_perimeter = perimeter_edges.sum(dim=(1, 2, 3))

        # 3. 破片碎裂指数 (C_frag)
        # 散粒裂缝/透明水滴等高频伪影，具有极其变态的 周长/面积 比！
        C_frag = E_perimeter / (E_root + 1e-5)

        # 当总能量极弱（纯土期），或极度破碎（大面积泥土裂缝爆发）时，双重触发因果截断
        flush_flags = (E_root < eta_energy) | (C_frag > theta_frag)

        return flush_flags  # 返回 [B] Boolean Tensor 触发 \emptyset Flush