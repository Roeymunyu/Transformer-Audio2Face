import os
import glob
import torch
import librosa
import numpy as np
import pandas as pd

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from scipy.interpolate import interp1d

# ==========================================
# 1. 参数与特征定义
# ==========================================
INPUT_DIR = "Dataset_Chunks"
OUTPUT_DIR = "Dataset_Tensors"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 强制要求的下面部 33 维特征
LOWER_FACE_BS = [
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

# 加载 HuggingFace 的 Wav2Vec2 预训练模型
print(" 正在加载 Wav2Vec2.0 模型 ")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
model.eval()

# ==========================================
# 2. 遍历处理每一个 Chunk
# ==========================================
wav_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.wav")))

for wav_path in wav_files:
    base_name = os.path.splitext(os.path.basename(wav_path))[0]
    csv_path = os.path.join(INPUT_DIR, f"{base_name}.csv")

    if not os.path.exists(csv_path):
        continue

    print(f"\n 正在处理: {base_name}")

    # --- A. 提取音频特征 (50 FPS) ---
    audio_input, sr = librosa.load(wav_path, sr=16000)

    # 转换为模型输入格式
    # inputs 包含的主要字段:
    # | `inputs.input_values`   | 模型实际输入的音频特征（张量）
    # | `inputs.attention_mask` | 哪些位置是真实数据 vs 填充（批量处理时用）
    inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        # 获取最后一层的隐状态 (Shape: [1, seq_len_50fps, 768])
        outputs = model(input_values)
        audio_features = outputs.last_hidden_state.squeeze(0).cpu().numpy()

    # --- B. 加载并清理面捕特征 (60 FPS) ---
    df = pd.read_csv(csv_path)

    # 容错检查：确保所有要求的列都在 CSV 中
    missing_cols = [col for col in LOWER_FACE_BS if col not in df.columns]
    if missing_cols:
        print(f"    警告: CSV 缺失列 {missing_cols}，跳过该文件")
        continue

    face_features = df[LOWER_FACE_BS].values  # (Shape: [seq_len_60fps, 33])

    # --- C. 最核心的对齐：线性插值 (重采样) ---
    len_audio_50fps = audio_features.shape[0]
    len_face_60fps = face_features.shape[0]

    print(f"  对齐前 -> 音频特征帧数: {len_audio_50fps} (约50Hz), 面捕帧数: {len_face_60fps} (60Hz)")

    # 构造两条绝对时间轴,这样采样点以分数形式精确表示
    time_audio = np.linspace(0, 1, len_audio_50fps)
    time_face = np.linspace(0, 1, len_face_60fps)

    # interp1d 为 768 每个维度都采用线性方式绘制了曲线，而 interp_func(time_face) 在曲线上以时间均分的方式采集了 600 个点
    # 创建插值函数 (沿着时间轴，对 768 维特征进行插值)
    interp_func = interp1d(
        time_audio, # 已知时间点 (500个)
        audio_features, # 已知数据 (500, 768)
        axis=0, # 沿着时间轴（第0维）插值
        kind='linear',
        fill_value="extrapolate" # 边界外推（防止越界）
    )

    # 将音频特征拉伸/压缩，强制对齐到面捕的帧数上
    aligned_audio_features = interp_func(time_face) # (600, 768)

    print(f"  对齐后 -> 音频特征: {aligned_audio_features.shape}, 面捕特征: {face_features.shape}")

    np.save(os.path.join(OUTPUT_DIR, f"{base_name}_audio.npy"), aligned_audio_features)
    np.save(os.path.join(OUTPUT_DIR, f"{base_name}_face.npy"), face_features)

print(f"\n 全部特征提取对齐完成,已保存至 {OUTPUT_DIR} 目录。")