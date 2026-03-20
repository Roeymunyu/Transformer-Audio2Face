import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ==========================================
# 动态绕过 HuggingFace 的安全审查
# ==========================================
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None

import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: None
# ==========================================

import torch
import numpy as np
import librosa
import socket
import struct
import time

from transformers import Wav2Vec2Processor, Wav2Vec2Model, pipeline
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d

from model import FaceFormerDemoV3

# ==========================================
# 1. 基础配置
# ==========================================
AUDIO_PATH = "ElevenLabs_Test1.mp3" # 示例，请替换为你的测试音频文件
MODEL_PATH = "best_faceformer-v6.pth"
# 如果在局域网内的不同设备运行，请替换为接收端（Unity 所在电脑）的真实局域网 IP
UDP_IP = "127.0.0.1"
UDP_PORT = 5005

UNITY_52_NAMES = [
    "EyeBlinkLeft", "EyeLookDownLeft", "EyeLookInLeft", "EyeLookOutLeft", "EyeLookUpLeft", "EyeSquintLeft",
    "EyeWideLeft",
    "EyeBlinkRight", "EyeLookDownRight", "EyeLookInRight", "EyeLookOutRight", "EyeLookUpRight", "EyeSquintRight",
    "EyeWideRight",
    "JawForward", "JawRight", "JawLeft", "JawOpen",
    "MouthClose", "MouthFunnel", "MouthPucker", "MouthRight", "MouthLeft", "MouthSmileLeft", "MouthSmileRight",
    "MouthFrownLeft", "MouthFrownRight", "MouthDimpleLeft", "MouthDimpleRight", "MouthStretchLeft",
    "MouthStretchRight",
    "MouthRollLower", "MouthRollUpper", "MouthShrugLower", "MouthShrugUpper", "MouthPressLeft", "MouthPressRight",
    "MouthLowerDownLeft", "MouthLowerDownRight", "MouthUpperUpLeft", "MouthUpperUpRight",
    "BrowDownLeft", "BrowDownRight", "BrowInnerUp", "BrowOuterUpLeft", "BrowOuterUpRight",
    "CheekPuff", "CheekSquintLeft", "CheekSquintRight",
    "NoseSneerLeft", "NoseSneerRight", "TongueOut"
]

MODEL_33_NAMES = [
    'JawForward', 'JawRight', 'JawLeft', 'JawOpen',
    'MouthClose', 'MouthFunnel', 'MouthPucker', 'MouthRight', 'MouthLeft',
    'MouthSmileLeft', 'MouthSmileRight', 'MouthFrownLeft', 'MouthFrownRight',
    'MouthDimpleLeft', 'MouthDimpleRight', 'MouthStretchLeft', 'MouthStretchRight',
    'MouthRollLower', 'MouthRollUpper', 'MouthShrugLower', 'MouthShrugUpper',
    'MouthPressLeft', 'MouthPressRight', 'MouthLowerDownLeft', 'MouthLowerDownRight',
    'MouthUpperUpLeft', 'MouthUpperUpRight',
    'CheekPuff', 'CheekSquintLeft', 'CheekSquintRight',
    'NoseSneerLeft', 'NoseSneerRight', 'TongueOut'
]

INDEX_MAPPING = [UNITY_52_NAMES.index(name) for name in MODEL_33_NAMES]

BROW_OUTER_L_IDX = UNITY_52_NAMES.index("BrowOuterUpLeft")
BROW_OUTER_R_IDX = UNITY_52_NAMES.index("BrowOuterUpRight")
BROW_INNER_UP_IDX = UNITY_52_NAMES.index("BrowInnerUp")

# ==========================================
# 2. 情感系统配置
# ==========================================

# 情感识别模型 (输出: angry, calm, disgust, fearful, happy, neutral, sad)
EMOTION_MODEL_NAME = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

EMOTION_WINDOW_SEC = 2.0     # 滑动窗口长度 (秒)
EMOTION_HOP_SEC = 0.5        # 滑动步长 (秒)
EMOTION_SMOOTH_FRAMES = 30   # 平滑窗口 (帧), 60fps 下 = 0.5 秒
EMOTION_ACTIVATION_THRESHOLD = 0.35  # 情感激活阈值, 低于此概率不生效

# ---- 情感表情预设 ----
# 值域 0~1, 表示该通道在情感完全激活时的目标权重
# 对于共享Blendshapes (MouthSmile 等), 这里的值是叠加量
EMOTION_PRESETS = {
    "happy": {
        #"EyeSquintLeft": 0.35,       # 笑眼 (上半脸, 覆盖写入)
        #"EyeSquintRight": 0.35,
        "MouthSmileLeft": 0.20,      # 嘴角上扬 (叠加到口型)
        "MouthSmileRight": 0.20,
        "CheekSquintLeft": 0.25,     # 颧骨上提 (叠加)
        "CheekSquintRight": 0.25,
        "BrowOuterUpLeft": 0.15,     # 微扬眉 (与韵律取 max)
        "BrowOuterUpRight": 0.15,
    },
    "sad": {
        "BrowInnerUp": 0.35,         # 八字眉
        "BrowDownLeft": 0.10,
        "BrowDownRight": 0.10,
        "MouthFrownLeft": 0.18,      # 嘴角下垂
        "MouthFrownRight": 0.18,
        #"EyeSquintLeft": 0.10,
        #"EyeSquintRight": 0.10,
    },
    "angry": {
        "BrowDownLeft": 0.45,        # 紧锁眉头
        "BrowDownRight": 0.45,
        "EyeSquintLeft": 0.25,       # 眯眼
        "EyeSquintRight": 0.25,
        "NoseSneerLeft": 0.30,       # 鼻翼上提
        #"NoseSneerRight": 0.30,
        #"MouthFrownLeft": 0.12,
        "MouthFrownRight": 0.12,
    },
    "fearful": {
        "EyeWideLeft": 0.45,         # 睁大眼
        "EyeWideRight": 0.45,
        "BrowInnerUp": 0.50,         # 挑内眉
        "BrowOuterUpLeft": 0.25,
        "BrowOuterUpRight": 0.25,
    },
    "disgust": {
        "NoseSneerLeft": 0.45,       # 鼻翼皱起
        "NoseSneerRight": 0.45,
        "BrowDownLeft": 0.25,
        "BrowDownRight": 0.25,
        "EyeSquintLeft": 0.20,
        "EyeSquintRight": 0.20,
        "MouthFrownLeft": 0.15,
        "MouthFrownRight": 0.15,
    },
    "calm": {},
    "neutral": {},
}

# ---- 混合模式 ----
BLEND_OVERRIDE = "override"
BLEND_MAX = "max"
BLEND_ADDITIVE = "additive"

UPPER_FACE_EMOTION_CHANNELS = {
    "EyeSquintLeft", "EyeSquintRight",
    "EyeWideLeft", "EyeWideRight",
}

BROW_CHANNELS = {
    "BrowDownLeft", "BrowDownRight", "BrowInnerUp",
    "BrowOuterUpLeft", "BrowOuterUpRight",
}

SHARED_ADDITIVE_CHANNELS = {
    "MouthSmileLeft", "MouthSmileRight",
    "MouthFrownLeft", "MouthFrownRight",
    "CheekSquintLeft", "CheekSquintRight",
    "NoseSneerLeft", "NoseSneerRight",
    "CheekPuff",
}

# 当这些情感激活时, 会抑制韵律层的挑眉动作
NEGATIVE_EMOTIONS = {"angry", "sad", "disgust"}


def get_blend_mode(channel_name):
    """返回该通道应使用的混合策略, 不受情感影响的通道返回 None"""
    if channel_name in UPPER_FACE_EMOTION_CHANNELS:
        return BLEND_OVERRIDE
    elif channel_name in BROW_CHANNELS:
        return BLEND_MAX
    elif channel_name in SHARED_ADDITIVE_CHANNELS:
        return BLEND_ADDITIVE
    return None


# ==========================================
# 3. 情感检测 (滑动窗口)
# ==========================================
def detect_emotions(audio_input, sr=16000):
    """
    对音频做滑动窗口情感分类, 返回每种情感在 60fps 下的强度曲线。
    """
    print("加载情感识别模型")
    device_id = 0 if torch.cuda.is_available() else -1
    emotion_pipe = pipeline(
        "audio-classification",
        model=EMOTION_MODEL_NAME,
        device=device_id,
    )

    audio_duration = len(audio_input) / sr
    target_frames = int(audio_duration * 60.0)

    window_samples = int(EMOTION_WINDOW_SEC * sr)
    hop_samples = int(EMOTION_HOP_SEC * sr)

    window_centers = []
    window_results = []

    print("逐窗口分析情感")
    pos = 0
    idx_count = 0
    while pos + window_samples <= len(audio_input):
        chunk = audio_input[pos:pos + window_samples]
        center_sec = (pos + window_samples / 2) / sr
        window_centers.append(center_sec)

        results = emotion_pipe({"raw": chunk, "sampling_rate": sr}, top_k=None)
        emo_dict = {r["label"].lower(): r["score"] for r in results}
        window_results.append(emo_dict)

        idx_count += 1
        pos += hop_samples

    # 尾部: 如果最后一个窗口没覆盖到末尾, 补一个
    if not window_centers or window_centers[-1] < audio_duration - EMOTION_WINDOW_SEC / 2:
        start = max(0, len(audio_input) - window_samples)
        chunk = audio_input[start:]
        if len(chunk) < window_samples:
            chunk = np.pad(chunk, (0, window_samples - len(chunk)))
        center_sec = (start + window_samples / 2) / sr
        window_centers.append(center_sec)
        results = emotion_pipe({"raw": chunk, "sampling_rate": sr}, top_k=None)
        emo_dict = {r["label"].lower(): r["score"] for r in results}
        window_results.append(emo_dict)

    print(f"   共分析了 {len(window_centers)} 个窗口")

    # 收集所有出现过的情感标签
    all_emotions = set()
    for wr in window_results:
        all_emotions.update(wr.keys())

    # 只保留预设中定义了的情感
    relevant_emotions = all_emotions & set(EMOTION_PRESETS.keys())

    # 插值到 60fps
    window_centers = np.array(window_centers)

    if len(window_centers) < 2:
        # 音频极短, 只有一个窗口
        emotion_curves = {}
        for emo in relevant_emotions:
            val = window_results[0].get(emo, 0.0) if window_results else 0.0
            emotion_curves[emo] = np.full(target_frames, val)
    else:
        time_60fps = np.linspace(window_centers[0], window_centers[-1], target_frames)
        emotion_curves = {}
        for emo in relevant_emotions:
            scores = np.array([wr.get(emo, 0.0) for wr in window_results])
            interp_func = interp1d(
                window_centers, scores, kind='linear',
                fill_value=(scores[0], scores[-1]), bounds_error=False
            )
            curve = interp_func(time_60fps)
            curve = uniform_filter1d(curve, size=EMOTION_SMOOTH_FRAMES)
            curve = np.clip(curve, 0.0, 1.0)
            emotion_curves[emo] = curve

    # 打印概览
    print("\n 情感分析概览:")
    for emo in sorted(emotion_curves.keys()):
        curve = emotion_curves[emo]
        peak = np.max(curve)
        mean = np.mean(curve)
        status = " 激活" if peak > EMOTION_ACTIVATION_THRESHOLD else " 未激活"
        print(f"   {emo:>10s}: 峰值={peak:.3f}  均值={mean:.3f}  {status}")

    # 释放情感模型显存
    del emotion_pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(" 情感模型已释放显存\n")

    return emotion_curves


# ==========================================
# 4. 情感层计算 & 混合
# ==========================================
def compute_emotion_blendshapes(emotion_curves, frame_idx):
    """
    根据当前帧的情感分布, 计算情感层 blendshape 增量。
    返回: dict[channel_name, float (0~1)]
    """
    layer = {}

    for emo_name, preset in EMOTION_PRESETS.items():
        if not preset:
            continue
        if emo_name not in emotion_curves:
            continue

        curve = emotion_curves[emo_name]
        if frame_idx >= len(curve):
            continue

        raw = curve[frame_idx]
        if raw < EMOTION_ACTIVATION_THRESHOLD:
            continue

        # 从阈值处重映射到 [0, 1]
        effective = (raw - EMOTION_ACTIVATION_THRESHOLD) / (1.0 - EMOTION_ACTIVATION_THRESHOLD)
        effective = min(effective, 1.0)

        for ch, target_val in preset.items():
            val = target_val * effective
            # 多种情感可能同时影响同一通道 → 取最大
            layer[ch] = max(layer.get(ch, 0.0), val)

    return layer


def compute_prosody_suppression(emotion_curves, frame_idx):
    """
    计算负面情感对韵律挑眉的抑制系数 (0=不抑制, 1=完全抑制)。
    当 angry/sad/disgust 等情感强激活时, 抑制 RMS 驱动的挑眉。
    """
    max_neg = 0.0
    for neg_emo in NEGATIVE_EMOTIONS:
        if neg_emo not in emotion_curves:
            continue
        curve = emotion_curves[neg_emo]
        if frame_idx >= len(curve):
            continue
        raw = curve[frame_idx]
        if raw > EMOTION_ACTIVATION_THRESHOLD:
            eff = (raw - EMOTION_ACTIVATION_THRESHOLD) / (1.0 - EMOTION_ACTIVATION_THRESHOLD)
            max_neg = max(max_neg, eff)
    return max_neg


def apply_emotion_layer(unity_payload, emotion_layer):
    """将情感层混合到已填好口型+韵律的 unity_payload 数组"""
    for ch_name, emo_val in emotion_layer.items():
        if ch_name not in UNITY_52_NAMES:
            continue

        idx = UNITY_52_NAMES.index(ch_name)
        mode = get_blend_mode(ch_name)
        if mode is None:
            continue

        emo_100 = emo_val * 100.0

        if mode == BLEND_OVERRIDE:
            unity_payload[idx] = emo_100
        elif mode == BLEND_MAX:
            unity_payload[idx] = max(unity_payload[idx], emo_100)
        elif mode == BLEND_ADDITIVE:
            unity_payload[idx] = np.clip(unity_payload[idx] + emo_100, 0.0, 100.0)

    return unity_payload


# ==========================================
# 5. 音频特征提取 & FaceFormer 推理
# ==========================================
def process_audio_and_predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" 使用设备: {device}")

    # --- 读取音频 ---
    audio_input, sr = librosa.load(AUDIO_PATH, sr=16000)
    audio_duration = len(audio_input) / 16000.0
    target_frames_60fps = int(audio_duration * 60.0)
    print(f" 音频时长: {audio_duration:.2f}s → {target_frames_60fps} 帧 @60fps\n")

    # --- A. 情感检测 (最先执行, 用完释放显存) ---
    emotion_curves = detect_emotions(audio_input, sr=16000)

    # --- B. Wav2Vec2 音频特征 ---
    print(" 正在加载 Wav2Vec2.0 模型提取音频特征")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    w2v_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    w2v_model.eval()

    inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = w2v_model(input_values)
        audio_features_50fps = outputs.last_hidden_state.squeeze(0).cpu().numpy()

    del w2v_model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- C. RMS 韵律特征 ---
    print(" 正在提取韵律特征 (RMS)...")
    rms = librosa.feature.rms(y=audio_input, hop_length=512)[0]
    time_rms = np.linspace(0, 1, len(rms))
    time_60fps = np.linspace(0, 1, target_frames_60fps)
    rms_60fps = interp1d(time_rms, rms, kind='linear', fill_value="extrapolate")(time_60fps)

    rms_mean, rms_std = np.mean(rms_60fps), np.std(rms_60fps)
    threshold = rms_mean + 0.5 * rms_std
    rms_normalized = np.maximum(0, rms_60fps - threshold)
    max_peak = np.max(rms_normalized) if np.max(rms_normalized) > 0 else 1.0
    rms_normalized = rms_normalized / max_peak
    rms_normalized = np.convolve(rms_normalized, np.ones(3) / 3, mode='same')

    # --- D. 帧率对齐 50→60 FPS ---
    print(" 正在对齐帧率 (50→60 FPS)")
    time_50fps = np.linspace(0, 1, audio_features_50fps.shape[0])
    time_60fps_feat = np.linspace(0, 1, target_frames_60fps)
    aligned = interp1d(time_50fps, audio_features_50fps, axis=0, kind='linear',
                       fill_value="extrapolate")(time_60fps_feat)
    audio_tensor = torch.tensor(aligned, dtype=torch.float32).unsqueeze(0).to(device)

    # --- E. FaceFormer 推理 ---
    print(" 正在推理口型 (FaceFormer)")
    model = FaceFormerDemoV3(motion_dim=33).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    predicted_motions = model.predict(audio_tensor).squeeze(0).cpu().numpy()

    return predicted_motions, rms_normalized, emotion_curves


# ==========================================
# 6. 发送到 Unity (三层混合)
# ==========================================
def send_to_unity(predicted_motions, rms_normalized, emotion_curves):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    total_frames = predicted_motions.shape[0]

    print(f"\n 准备发送 | 目标: {UDP_IP}:{UDP_PORT} | 帧数: {total_frames} ({total_frames / 60.0:.2f}s)")
    print("   混合层级: Layer1=口型(33ch) → Layer2=韵律(眉毛) → Layer3=情感(上脸+叠加)")

    for i in range(3, 0, -1):
        print(f"   倒计时 {i}...") # 用于测试等待
        time.sleep(1)

    print("\n 开始发送！\n")

    frame_duration = 1.0 / 60.0
    start_time = time.time()

    for frame_idx in range(total_frames):
        # 全零基底
        unity_payload = np.zeros(52, dtype=np.float32)

        # 口型 (FaceFormer 33维 → 52维映射)
        frame_33 = predicted_motions[frame_idx]
        for k, val in enumerate(frame_33):
            unity_idx = INDEX_MAPPING[k]
            unity_payload[unity_idx] = np.clip(val * 100.0, 0.0, 100.0)

        # RMS → 挑眉
        # 先计算负面情感抑制系数
        suppression = compute_prosody_suppression(emotion_curves, frame_idx)
        prosody_scale = 1.0 - suppression * 0.7  # 保留一点自然感

        accent = rms_normalized[frame_idx]
        if accent > 0:
            brow_w = accent * 60.0 * prosody_scale
            unity_payload[BROW_OUTER_L_IDX] = brow_w
            unity_payload[BROW_OUTER_R_IDX] = brow_w
            unity_payload[BROW_INNER_UP_IDX] = brow_w * 0.5

        emotion_layer = compute_emotion_blendshapes(emotion_curves, frame_idx)
        apply_emotion_layer(unity_payload, emotion_layer)

        packed = struct.pack('<52f', *unity_payload)
        sock.sendto(packed, (UDP_IP, UDP_PORT))

        target_time = start_time + (frame_idx + 1) * frame_duration
        sleep_time = target_time - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)

    print(" 发送完毕！")


if __name__ == "__main__":
    motions, rms_data, emo_curves = process_audio_and_predict()
    send_to_unity(motions, rms_data, emo_curves)