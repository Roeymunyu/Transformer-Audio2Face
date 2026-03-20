import torch
import torch.nn as nn
import math


def init_biased_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + \
                   get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]

    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1, period).view(-1) // period
    bias = -torch.flip(bias, dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i + 1] = bias[-(i + 1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask


# ======= 改动点 1：窗口从 1 扩大到 4 =======
def enc_dec_mask(device, T, S, window=4):
    """
    每个 motion 帧关注 ±window 个 audio 帧
    window=1 → 3帧 ≈ 60ms （之前）
    window=4 → 9帧 ≈ 180ms（现在）→ 覆盖完整音素
    """
    mask = torch.ones(T, S)
    for i in range(T):
        start = max(0, i - window)
        end = min(S, i + window + 1)
        mask[i, start:end] = 0
    return (mask == 1).to(device=device)


class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=6000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        pe = pe.repeat(1, (max_seq_len // period) + 1, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class FaceFormerDemoV3(nn.Module):
    def __init__(self, audio_dim=768, motion_dim=33, feature_dim=256, num_heads=4, period=30):
        super().__init__()
        self.motion_dim = motion_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.audio_feature_map = nn.Linear(audio_dim, feature_dim)

        # ======== 改动点 2：残差时序卷积 ======================
        # Wav2Vec 特征是全局的（每帧已看过全部音频）
        # 1D 卷积提取局部音素过渡模式（如 a→o 的频谱梯度）
        # 残差连接保留全局信息
        self.audio_temporal_conv = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=5, padding=2),
        )

        self.vertice_map = nn.Linear(motion_dim, feature_dim)
        self.PPE = PeriodicPositionalEncoding(feature_dim, period=period, max_seq_len=6000)
        self.biased_mask = init_biased_mask(n_head=num_heads, max_seq_len=6000, period=period)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim, nhead=num_heads,
            dim_feedforward=2 * feature_dim, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        self.vertice_map_r = nn.Linear(feature_dim, motion_dim)
        nn.init.constant_(self.vertice_map_r.weight, 0)
        nn.init.constant_(self.vertice_map_r.bias, 0)

    def _encode_audio(self, hidden_states):
        """音频编码：线性映射 + 残差时序卷积"""
        audio = self.audio_feature_map(hidden_states)
        audio = audio + self.audio_temporal_conv(audio.transpose(1, 2)).transpose(1, 2)
        return audio

    def forward(self, hidden_states, target_motions):
        batch_size, seq_len, _ = target_motions.size()
        audio_memory = self._encode_audio(hidden_states)

        template = torch.zeros((batch_size, 1, self.motion_dim), device=self.device)
        motion_input = torch.cat((template, target_motions[:, :-1, :]), dim=1)
        motion_input = self.PPE(self.vertice_map(motion_input))

        tgt_mask = self.biased_mask[:, :seq_len, :seq_len].clone().detach().to(self.device)
        nh = tgt_mask.size(0)
        tgt_mask = tgt_mask.unsqueeze(0).expand(batch_size, -1, -1, -1).reshape(batch_size * nh, seq_len, seq_len)
        memory_mask = enc_dec_mask(self.device, seq_len, audio_memory.size(1))

        decoder_out = self.transformer_decoder(
            tgt=motion_input, memory=audio_memory,
            tgt_mask=tgt_mask, memory_mask=memory_mask
        )
        return torch.clamp(self.vertice_map_r(decoder_out), 0.0, 1.0)

    def predict(self, hidden_states):
        self.eval()
        batch_size, seq_len, _ = hidden_states.size()
        audio_memory = self._encode_audio(hidden_states)

        template = torch.zeros((batch_size, 1, self.motion_dim), device=self.device)
        generated = template

        with torch.no_grad():
            for i in range(seq_len):
                motion_input = self.PPE(self.vertice_map(generated))
                cl = motion_input.size(1)

                tgt_mask = self.biased_mask[:, :cl, :cl].clone().detach().to(self.device)
                nh = tgt_mask.size(0)
                tgt_mask = tgt_mask.unsqueeze(0).expand(batch_size, -1, -1, -1).reshape(batch_size * nh, cl, cl)
                memory_mask = enc_dec_mask(self.device, cl, seq_len)

                decoder_out = self.transformer_decoder(
                    tgt=motion_input, memory=audio_memory,
                    tgt_mask=tgt_mask, memory_mask=memory_mask
                )
                new_frame = torch.clamp(self.vertice_map_r(decoder_out)[:, -1:, :], 0.0, 1.0)
                generated = torch.cat((generated, new_frame), dim=1)

        self.train()
        return generated[:, 1:, :]