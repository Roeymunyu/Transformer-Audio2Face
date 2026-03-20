import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# 1. 加载预训练的 Wav2Vec2 模型 (Base版本，960小时英语训练集，对中文口型同样有效)
print("正在加载 Wav2Vec2 模型，初次运行需下载，请稍候...")
model_name = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name)
print("✅ 模型加载完成！")

# 2. 加载你的测试音频
# Wav2Vec2 严格要求 16000 Hz 的采样率，librosa 会帮我们自动重采样
audio_path = "test_voice.wav"
print(f"正在读取音频: {audio_path}")
speech_array, sampling_rate = librosa.load(audio_path, sr=16000)

# 3. 预处理：将一维的音频波形转为模型需要的输入格式
input_values = processor(speech_array, return_tensors="pt", sampling_rate=16000).input_values

# 4. 提取特征 (AI 推理环节)
print("正在提取语音特征...")
with torch.no_grad(): # 推理模式，不计算梯度，节省内存
    outputs = model(input_values)
    # 我们需要的是最后一层的隐藏状态 (Hidden States)
    hidden_states = outputs.last_hidden_state

# 5. 见证奇迹的时刻
print("-" * 30)
print(f"🎤 音频总时长: {len(speech_array) / 16000:.2f} 秒")
print(f"📦 提取的特征张量形状 (Shape): {hidden_states.shape}")
print("-" * 30)