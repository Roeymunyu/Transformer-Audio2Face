import os
import glob
import numpy as np
from scipy.signal import savgol_filter

INPUT_DIR = "Dataset_Tensors"

# 找到所有面捕数据张量
face_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*_face.npy")))

print(f" 准备对 {len(face_files)} 个面部张量进行 Savitzky-Golay 平滑处理...")

for file_path in face_files:
    # 加载原始 33 维面捕数据, shape: [seq_len, 33]
    face_data = np.load(file_path)

    # -------------------------------------------------------------
    # 核心：Savitzky-Golay 滤波
    # window_length=7: 每次看前后共 7 帧
    # polyorder=3: 使用 3 次多项式拟合曲线
    # axis=0: 沿着时间轴 (seq_len) 对每一列 (每一维 Blendshape) 独立平滑
    # -------------------------------------------------------------
    # 确保序列长度至少大于窗口长度，否则直接跳过或用更小窗口
    if face_data.shape[0] < 7:
        smoothed_data = face_data
    else:
        smoothed_data = savgol_filter(face_data, window_length=7, polyorder=3, axis=0)

    # 防止平滑后出现负数或超过1的值 (ARKit Blendshape 的范围严格在 0.0 ~ 1.0 之间)
    smoothed_data = np.clip(smoothed_data, 0.0, 1.0)

    # 直接覆盖原文件，或者你可以保存为 *_face_smooth.npy 以防万一
    # 这里我们选择安全覆盖，因为已经准备好直接喂给 Dataset 了
    np.save(file_path, smoothed_data)

print(" 全部数据平滑完毕")