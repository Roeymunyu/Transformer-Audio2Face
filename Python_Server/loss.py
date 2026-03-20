import torch
import torch.nn as nn


class FaceFormerLoss(nn.Module):
    """
    v6 损失函数：在 v5 基础上新增
    - Pearson 相关损失：直接惩罚时序模式不匹配（解决负相关/零相关问题）
    - 偏置正则化：显式拉回静息态维度的均值偏移
    - 表情维度降权：释放模型容量给口型
    """

    def __init__(self, device, motion_dim=33,
                 pos_weight=1.0, vel_weight=10.0,
                 contrastive_weight=0.05, var_weight=0.3,
                 corr_weight=0.03, bias_weight=0.3):
        super().__init__()
        self.device = device
        self.motion_dim = motion_dim

        self.pos_weight = pos_weight
        self.vel_weight = vel_weight
        self.contrastive_weight = contrastive_weight
        self.var_weight = var_weight
        self.corr_weight = corr_weight
        self.bias_weight = bias_weight

        # ==========================================
        # 位置损失权重（L2用）
        # ==========================================
        self.loss_weights = torch.ones(motion_dim, device=device)

        # --- 核心口型（最高优先） ---
        self.loss_weights[3] = 4.0    # JawOpen
        self.loss_weights[4] = 4.0    # MouthClose
        self.loss_weights[5] = 3.0    # MouthFunnel
        self.loss_weights[6] = 3.0    # MouthPucker

        # --- 唇部细节 ---
        self.loss_weights[19] = 1.5   # MouthShrugLower
        self.loss_weights[20] = 1.5   # MouthShrugUpper

        # --- 偏置灾区（适度加权，交给corr_loss处理时序） ---
        self.loss_weights[17] = 2.0   # MouthRollLower
        self.loss_weights[18] = 2.0   # MouthRollUpper
        self.loss_weights[0] = 2.0    # JawForward

        # --- 降权：非口型维度（释放模型容量） ---
        self.loss_weights[9] = 0.3    # MouthSmileLeft
        self.loss_weights[10] = 0.3   # MouthSmileRight
        self.loss_weights[11] = 0.3   # MouthFrownLeft
        self.loss_weights[12] = 0.3   # MouthFrownRight
        # NoseSneer/MouthDimple 依然为0.5，保持不变
        self.loss_weights[13] = 0.5
        self.loss_weights[14] = 0.5
        self.loss_weights[30] = 0.5
        self.loss_weights[31] = 0.5

        # ==========================================
        # 相关损失权重（只对音素驱动维度生效）
        # 非音素维度设为 0 → 不浪费梯度
        # ==========================================
        self.corr_weights = torch.zeros(motion_dim, device=device)

        # 核心发音器官（最高相关权重）
        self.corr_weights[3] = 2.0    # JawOpen
        self.corr_weights[4] = 2.0    # MouthClose
        self.corr_weights[5] = 2.5    # MouthFunnel  重点修一修
        self.corr_weights[6] = 2.5    # MouthPucker  重点修一修

        # 唇部运动
        self.corr_weights[17] = 2.0   # MouthRollLower 重点修一修
        self.corr_weights[18] = 1.5   # MouthRollUpper
        self.corr_weights[19] = 1.0   # MouthShrugLower
        self.corr_weights[20] = 1.0   # MouthShrugUpper

        # 嘴唇上下运动
        self.corr_weights[23] = 1.5   # MouthLowerDownLeft
        self.corr_weights[24] = 1.5   # MouthLowerDownRight
        self.corr_weights[25] = 1.0   # MouthUpperUpLeft
        self.corr_weights[26] = 1.0   # MouthUpperUpRight

        # 嘴唇伸展
        self.corr_weights[15] = 1.0   # MouthStretchLeft
        self.corr_weights[16] = 1.0   # MouthStretchRight

        # MouthSmile/Frown/NoseSneer/MouthDimple → 保持 0

        # ==========================================
        # 偏置正则化目标维度
        # 这些维度在GT中均值 < 0.1，容易被模型预测为偏高常数
        # ==========================================
        self.rest_state_dims = [2, 17, 18]  # JawForward, MouthRollLower, MouthRollUpper

    def _correlation_loss(self, pred, target):
        """
        Pearson 相关损失：1 - r（按维度，加权平均）
        - r = 1.0 → loss = 0（完美相关）
        - r = 0.0 → loss = 1.0（无相关，惩罚"输出常数"策略）
        - r = -0.13 → loss = 1.13（负相关，强惩罚）
        """
        # [B, T, D]
        pred_m = pred.mean(dim=1, keepdim=True)
        tgt_m = target.mean(dim=1, keepdim=True)
        pred_c = pred - pred_m
        tgt_c = target - tgt_m

        numerator = (pred_c * tgt_c).sum(dim=1)
        # 修复：eps 放在 sqrt 内部，避免 sqrt(0) 的梯度为 inf
        pred_std = (pred_c.pow(2).sum(dim=1) + 1e-8).sqrt()
        tgt_std = (tgt_c.pow(2).sum(dim=1) + 1e-8).sqrt()
        denominator = pred_std * tgt_std    # [B, D]

        corr = numerator / denominator
        loss = 1.0 - corr

        # 加权：只对 corr_weights > 0 的维度施加相关损失
        weighted = loss * self.corr_weights.unsqueeze(0)

        # 归一化：除以权重总和 × batch，得到可解释的标量
        total_weight = self.corr_weights.sum() + 1e-8
        return weighted.sum() / (total_weight * pred.size(0))

    def _bias_loss(self, pred, target):
        """
        偏置正则化：惩罚静息态维度的均值偏移
        对 JawForward, MouthRollLower/Upper 这类在GT中接近0的维度，
        显式拉回预测均值
        """
        if not self.rest_state_dims:
            return torch.tensor(0.0, device=self.device)
        dims = self.rest_state_dims
        pred_mean = pred[:, :, dims].mean(dim=1)      # [B, num_dims]
        tgt_mean = target[:, :, dims].mean(dim=1)      # [B, num_dims]
        return (pred_mean - tgt_mean).abs().mean()

    def forward(self, predictions, targets):
        """
        返回: total_loss, pos_loss, vel_loss, contra_loss, var_loss, corr_loss, bias_loss
        """
        # 1. 位置损失 (L2)
        pos_loss = torch.mean((predictions - targets) ** 2 * self.loss_weights)

        # 2. 速度损失 (L2)
        pred_vel = predictions[:, 1:, :] - predictions[:, :-1, :]
        tgt_vel = targets[:, 1:, :] - targets[:, :-1, :]
        vel_loss = torch.mean((pred_vel - tgt_vel) ** 2 * self.loss_weights)

        # 3. 对比损失 (JawOpen 开闭区分)
        pj = predictions[:, :, 3]
        tj = targets[:, :, 3]
        open_mask = tj > 0.4
        close_mask = (targets[:, :, 4] > 0.3) | (tj < 0.1)
        pull = torch.mean((pj[open_mask] - tj[open_mask]) ** 2) if open_mask.any() \
            else torch.tensor(0.0, device=self.device)
        push = torch.mean((pj[close_mask] - tj[close_mask]) ** 2) if close_mask.any() \
            else torch.tensor(0.0, device=self.device)
        contrastive_loss = pull + push

        # 4. 方差损失 (单侧惩罚)
        pred_std = torch.std(predictions, dim=1)
        tgt_std = torch.std(targets, dim=1)
        var_deficit = torch.clamp(tgt_std - pred_std, min=0.0)
        var_loss = torch.mean(var_deficit * self.loss_weights.unsqueeze(0))

        # 5. 相关损失
        corr_loss = self._correlation_loss(predictions, targets)

        # 6. 偏置正则化
        bias_loss = self._bias_loss(predictions, targets)

        # 总损失
        total_loss = (self.pos_weight * pos_loss
                      + self.vel_weight * vel_loss
                      + self.contrastive_weight * contrastive_loss
                      + self.var_weight * var_loss
                      + self.corr_weight * corr_loss
                      + self.bias_weight * bias_loss)

        return total_loss, pos_loss, vel_loss, contrastive_loss, var_loss, corr_loss, bias_loss