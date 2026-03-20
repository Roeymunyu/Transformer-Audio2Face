import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset

class AudioFaceDataset(Dataset):
    def __init__(self, data_dir, seq_len=120, stride=30,is_train=True,split_ratio=0.9):
        self.seq_len = seq_len
        self.windows = []
        face_files = sorted(glob.glob(os.path.join(data_dir, "*_face_smooth.npy")))

        # 划分数据集
        split_idx = int(len(face_files) * split_ratio)
        if is_train:
            face_files = face_files[:split_idx]
            print(f" 构建 [训练集] (90%): 分配了 {len(face_files)} 个源文件")
        else:
            face_files = face_files[split_idx:]
            print(f" 构建 [测试集] (10%): 分配了 {len(face_files)} 个源文件")

        for face_path in face_files:
            audio_path = face_path.replace("_face_smooth.npy", "_audio.npy")
            if not os.path.exists(audio_path):
                continue
            try:
                total_frames = np.load(face_path, mmap_mode='r').shape[0]
            except:
                continue

            if total_frames < self.seq_len:
                continue

            for start_idx in range(0, total_frames - self.seq_len + 1, stride):
                end_idx = start_idx + self.seq_len
                self.windows.append({
                    "audio": audio_path,
                    "face": face_path,
                    "start": start_idx,
                    "end": end_idx
                })
        mode = "Train" if is_train else "Test"
        print(f"{mode} Dataset 加载完成: 发现 {len(self.windows)} 个有效 Sequence (长度={seq_len})")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        info = self.windows[idx]
        audio_data = np.load(info["audio"])[info["start"]:info["end"]]
        face_data = np.load(info["face"])[info["start"]:info["end"]]

        return torch.tensor(audio_data, dtype=torch.float32), torch.tensor(face_data, dtype=torch.float32)