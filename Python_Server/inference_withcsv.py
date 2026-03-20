import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import numpy as np
import librosa
import csv
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from scipy.interpolate import interp1d

from model import FaceFormerDemoV3

# ==========================================
# 1. 配置参数与映射关系
# ==========================================
AUDIO_PATH = "MySlate_3_iPhone.mov" # 请替换为你的测试文件
MODEL_PATH = "best_faceformer-v6.pth"
# 如果在局域网内的不同设备运行，请替换为接收端（Unity 所在电脑）的真实局域网 IP
UDP_IP = "127.0.0.1"
UDP_PORT = 5005

# Unity 期望的完整 52 维 Blendshapes 顺序
UNITY_52_NAMES = [
    "EyeBlinkLeft", "EyeLookDownLeft", "EyeLookInLeft", "EyeLookOutLeft", "EyeLookUpLeft", "EyeSquintLeft",
    "EyeWideLeft",
    "EyeBlinkRight", "EyeLookDownRight", "EyeLookInRight", "EyeLookOutRight", "EyeLookUpRight", "EyeSquintRight",
    "EyeWideRight",
    "JawForward", "JawRight", "JawLeft", "JawOpen",
    "MouthClose", "MouthFunnel", "MouthPucker", "MouthRight", "MouthLeft", "MouthSmileLeft", "MouthSmileRight",
    "MouthFrownLeft", "MouthFrownRight", "MouthDimpleLeft", "MouthDimpleRight", "MouthStretchLeft", "MouthStretchRight",
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


def process_audio_and_predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" 使用设备: {device}")

    # --- 1. 提取 Wav2Vec2 音频特征 ---
    print(" 正在加载 Wav2Vec2.0 模型提取音频特征")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    w2v_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    w2v_model.eval()

    audio_input, sr = librosa.load(AUDIO_PATH, sr=16000)
    inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = w2v_model(input_values)
        audio_features_50fps = outputs.last_hidden_state.squeeze(0).cpu().numpy()

    print(" 正在对齐音频特征帧率 (50FPS -> 60FPS) ")
    audio_duration = len(audio_input) / 16000.0
    target_frames_60fps = int(audio_duration * 60.0)

    time_50fps = np.linspace(0, 1, audio_features_50fps.shape[0])
    time_60fps = np.linspace(0, 1, target_frames_60fps)

    interp_func = interp1d(time_50fps, audio_features_50fps, axis=0, kind='linear', fill_value="extrapolate")
    aligned_audio_features = interp_func(time_60fps)

    audio_tensor = torch.tensor(aligned_audio_features, dtype=torch.float32).unsqueeze(0).to(
        device)  # Shape: [1, seq_len, 768]

    print("正在加载 FaceFormer 权重进行自回归推理 ")
    model = FaceFormerDemoV3(motion_dim=33).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    predicted_motions = model.predict(audio_tensor)  # Shape: [1, seq_len, 33]
    predicted_motions = predicted_motions.squeeze(0).cpu().numpy()

    return predicted_motions


def save_to_csv(predicted_motions, output_csv="predicted_blendshapes.csv"):
    """
    新增功能：将预测结果按指定格式保存到 CSV 中（未乘以 100 之前的幅度）
    """
    print(f"\n正在保存推断结果到 CSV: {output_csv}")

    fps = 60
    total_frames = predicted_motions.shape[0]

    # 构造表头：2个前缀信息 + 52个Blendshapes + 9个头部和眼部旋转参数
    cols = ['Timecode', 'BlendshapeCount'] + UNITY_52_NAMES + [
        'HeadYaw', 'HeadPitch', 'HeadRoll',
        'LeftEyeYaw', 'LeftEyePitch', 'LeftEyeRoll',
        'RightEyeYaw', 'RightEyePitch', 'RightEyeRoll'
    ]

    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(cols)  # 写入表头

        for frame_idx in range(total_frames):
            # 1. 构造时间戳，模拟 60FPS
            total_seconds = frame_idx / fps
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = int(total_seconds % 60)
            frames = frame_idx % fps
            milliseconds = int((total_seconds * 1000) % 1000)
            timecode = f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}.{milliseconds:03d}"

            # 2. 构造 52 维数据数组
            unity_payload = np.zeros(52, dtype=np.float32)
            frame_data_33 = predicted_motions[frame_idx]
            for idx, val in enumerate(frame_data_33):
                unity_idx = INDEX_MAPPING[idx]
                unity_payload[unity_idx] = val

            # 3. 补齐额外的 9 个头部/眼球旋转（置 0.0）
            extra_cols = [0.0] * 9

            # 4. 拼接写入单行数据：[时间戳, Blendshape总数(61), ...52个Blendshape幅度, ...9个旋转值]
            row = [timecode, 61] + unity_payload.tolist() + extra_cols
            writer.writerow(row)

    print(" CSV 文件保存成功！ ")


if __name__ == "__main__":
    motions = process_audio_and_predict()
    save_to_csv(motions, output_csv="predicted_blendshapes.csv")
