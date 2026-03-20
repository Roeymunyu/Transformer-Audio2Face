import os
import librosa
import pandas as pd
import soundfile as sf
import math

# ==========================================
# 1. 严格配置参数 (Transformer 数据标准)
# ==========================================
# 请填入实际的文件名， 以下为示例
file_pairs = [
    ("MySlate_10_iPhone.mov", "MySlate_10_iPhone.csv"),
    ("MySlate_11_iPhone.mov", "MySlate_11_iPhone.csv"),
    ("MySlate_12_iPhone.mov", "MySlate_12_iPhone.csv"),
]

OUTPUT_DIR = "Dataset_Chunks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SR = 16000  # 标准采样率
FPS = 60.0  # ARKit 标准帧率
TOP_DB = 30  # 静音检测阈值
MIN_DURATION = 3.0
MAX_DURATION = 15.0

# [新增参数]
PAD_S = 0.25  # 切片首尾各保留 250ms，保护面部预备发音和收尾的物理动作
MAX_SILENCE_GAP = 0.5  # 如果两句发音的停顿低于 0.5 秒则合并，大于则强行切断，避免 Chunk 内包含长段死静音


# ==========================================
# 辅助函数: 解析 Live Link Face 的 Timecode
# ==========================================
def parse_arkit_timecode(tc_str):
    # 示例格式: "04:43:58:53.495" -> HH:MM:SS:FF.subframes
    parts = str(tc_str).strip().split(':')
    if len(parts) < 4:
        return 0.0
    h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    frames = float(parts[3])
    # 转换为秒数 (ARKit 基于 60.0 FPS)
    return h * 3600 + m * 60 + s + frames / 60.0


# ==========================================
# 2. 静音切分与保存逻辑
# ==========================================
chunk_index = 0

for mov_path, csv_path in file_pairs:
    print(f"\n 正在处理文件: {mov_path} & {csv_path}")

    # 1. 读取音频
    print("  正在加载音频波形")
    speech_array, _ = librosa.load(mov_path, sr=SR)
    total_audio_duration = len(speech_array) / SR

    # 2. 读取并构建绝对时间轴
    print("   正在加载 CSV 数据并构建绝对物理时间轴")
    df = pd.read_csv(csv_path)

    # 计算每一行的绝对时间并转换为相对时长
    df['TimeSeconds'] = df['Timecode'].apply(parse_arkit_timecode)
    df['TimeSeconds'] = df['TimeSeconds'] - df['TimeSeconds'].iloc[0]
    total_csv_duration = df['TimeSeconds'].iloc[-1]

    print(f"   对齐检查 -> 音频: {total_audio_duration:.2f}s | 视频: {total_csv_duration:.2f}s")

    # 3. 执行改进版的 VAD 合并逻辑
    print("   在执行 VAD 智能静音切分")
    non_mute_intervals = librosa.effects.split(speech_array, top_db=TOP_DB)

    chunks = []
    if len(non_mute_intervals) > 0:
        current_start = non_mute_intervals[0][0]
        current_end = non_mute_intervals[0][1]

        for interval in non_mute_intervals[1:]:
            start, end = interval
            gap_s = (start - current_end) / SR
            duration_if_merged = (end - current_start) / SR

            # 仅当间隔极短且未超时，才进行自然合并
            if gap_s <= MAX_SILENCE_GAP and duration_if_merged <= MAX_DURATION:
                current_end = end
            else:
                # 结算当前 chunk
                if (current_end - current_start) / SR >= MIN_DURATION:
                    chunks.append((current_start, current_end))
                # 开启新 chunk
                current_start = start
                current_end = end

        # 结算最后一个 chunk
        if (current_end - current_start) / SR >= MIN_DURATION:
            chunks.append((current_start, current_end))

    # 4. 执行物理切分，追加边界缓冲，基于“时间”而非“行号”去提取 CSV
    valid_chunks_count = 0
    pad_samples = int(PAD_S * SR)

    for start_idx, end_idx in chunks:
        # [极关键] 增加 Padding，确保完整的口型准备动作
        chunk_start_idx = max(0, start_idx - pad_samples)
        chunk_end_idx = min(len(speech_array), end_idx + pad_samples)

        start_time = chunk_start_idx / SR
        end_time = chunk_end_idx / SR

        chunk_audio = speech_array[chunk_start_idx:chunk_end_idx]

        # 彻底抛弃 iloc 行号索引，使用绝对物理时间过滤
        chunk_df = df[(df['TimeSeconds'] >= start_time) & (df['TimeSeconds'] <= end_time)]

        # 过滤异常：容忍一定程度的掉帧，但如果缺失太多则抛弃
        expected_frames = (end_time - start_time) * FPS
        if len(chunk_df) < expected_frames * 0.85:
            continue

        chunk_name = f"chunk_{chunk_index:04d}"

        # 保存音频
        sf.write(os.path.join(OUTPUT_DIR, f"{chunk_name}.wav"), chunk_audio, SR)

        # 移除临时的时间计算列并保存 CSV
        chunk_df.drop(columns=['TimeSeconds']).to_csv(os.path.join(OUTPUT_DIR, f"{chunk_name}.csv"), index=False)

        chunk_index += 1
        valid_chunks_count += 1

    print(f"  成功过滤并对齐出 {valid_chunks_count} 个数据片段！")

print(f"\n 总共生成了 {chunk_index} 组面捕对，保存在 {OUTPUT_DIR} 目录下。")