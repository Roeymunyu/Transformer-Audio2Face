import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class AudioFaceDataset(Dataset):
    def __init__(self, data_dir, seq_len=120, stride=30, is_train=True):
        """
        参数:
        data_dir: 存放 .npy 文件的目录
        seq_len: 序列长度 (默认 120 帧，约 2 秒 @ 60FPS)
        stride: 滑动窗口的步长 (默认 30 帧，约 0.5 秒)
        is_train: 区分训练集和验证集标志
        """
        self.seq_len = seq_len
        self.windows = []

        # 获取所有平滑后的面捕数据和对应的音频数据
        face_files = sorted(glob.glob(os.path.join(data_dir, "*_face_smooth.npy")))

        # 数据集划分 (90% 训练, 10% 验证)
        split_idx = int(len(face_files) * 0.9)
        face_files = face_files[:split_idx] if is_train else face_files[split_idx:]

        for face_path in face_files:
            audio_path = face_path.replace("_face_smooth.npy", "_audio.npy")
            if not os.path.exists(audio_path):
                continue

            # 只需要知道文件长度来划分窗口，不需要将数据全部读入内存
            # 使用 mmap_mode='r' 仅读取 numpy header
            try:
                face_shape = np.load(face_path, mmap_mode='r').shape
                total_frames = face_shape[0]
            except Exception as e:
                print(f"读取文件形状出错: {face_path}, {e}")
                continue

            # 如果单个 chunk 长度小于 seq_len，直接跳过
            if total_frames < self.seq_len:
                continue

            # 计算当前文件可以提取多少个滑动窗口
            for start_idx in range(0, total_frames - self.seq_len + 1, stride):
                end_idx = start_idx + self.seq_len
                self.windows.append({
                    "audio_path": audio_path,
                    "face_path": face_path,
                    "start_idx": start_idx,
                    "end_idx": end_idx
                })

        print(f"构建 Dataset 完成 (is_train={is_train}): 共 {len(self.windows)} 个有效滑动窗口。")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window_info = self.windows[idx]

        # 从硬盘读取特定的切片范围
        # 使用 np.load 配合高级索引
        audio_data = np.load(window_info["audio_path"])[window_info["start_idx"]:window_info["end_idx"]]
        face_data = np.load(window_info["face_path"])[window_info["start_idx"]:window_info["end_idx"]]

        # 转换为 PyTorch Tensor
        audio_tensor = torch.tensor(audio_data, dtype=torch.float32)  # [seq_len, 768]
        face_tensor = torch.tensor(face_data, dtype=torch.float32)  # [seq_len, 33]

        return audio_tensor, face_tensor