import torch
from torch.utils.data import DataLoader

from dataset import AudioFaceDataset
from model import FaceFormerDemoV3
from loss import FaceFormerLoss


def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" 正在加载测试集进行模型评估")

    # 参数配置
    DATA_DIR = "Dataset_Tensors"
    MOTION_DIM = 33
    SEQ_LEN = 120
    MODEL_PATH = "best_faceformer-v6.pth"

    test_dataset = AudioFaceDataset(DATA_DIR, seq_len=SEQ_LEN, stride=30, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 加载模型
    model = FaceFormerDemoV3(motion_dim=MOTION_DIM).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f" 成功加载最优模型权重: {MODEL_PATH}")
    except Exception as e:
        print(f" 模型加载失败: {e}")
        return

    model.eval()
    criterion = FaceFormerLoss(device=device, motion_dim=MOTION_DIM)

    # 初始化下指标
    metrics = {
        'tf_loss': 0.0,
        'ar_loss': 0.0,
        'ar_pos': 0.0,
        'ar_vel': 0.0,
        'ar_contra': 0.0,
        'ar_var': 0.0,
        'ar_corr': 0.0,
        'ar_bias': 0.0
    }

    print("\n 开始双轨测试 (Teacher-Forcing vs Autoregressive)")
    with torch.no_grad():
        for batch_idx, (audio_batch, face_batch) in enumerate(test_loader):
            audio_batch = audio_batch.to(device)
            face_batch = face_batch.to(device)

            # ==========================================
            # 1. Teacher-Forcing 测试 (理论上限)
            # ==========================================
            tf_predictions = model(audio_batch, face_batch)
            # 必须解包 7 个值以匹配 FaceFormerLoss 的返回
            tf_results = criterion(tf_predictions, face_batch)
            metrics['tf_loss'] += tf_results[0].item()

            # ==========================================
            # 2. 自回归逐帧测试 (模拟真实表现)
            # ==========================================
            ar_predictions = model.predict(audio_batch)

            # 解包
            ar_total, ar_pos, ar_vel, ar_contra, ar_var, ar_corr, ar_bias = criterion(ar_predictions, face_batch)

            metrics['ar_loss'] += ar_total.item()
            metrics['ar_pos'] += ar_pos.item()
            metrics['ar_vel'] += ar_vel.item()
            metrics['ar_contra'] += ar_contra.item()
            metrics['ar_var'] += ar_var.item()
            metrics['ar_corr'] += ar_corr.item()
            metrics['ar_bias'] += ar_bias.item()

            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(test_loader):
                print(f"   Batch [{batch_idx + 1:2d}/{len(test_loader)}] | "
                      f"TF Loss: {tf_results[0].item():.5f} | "
                      f"AR Pos: {ar_pos.item():.5f} | AR Corr: {ar_corr.item():.4f}")

    n = len(test_loader)
    for key in metrics:
        metrics[key] /= n

    print("\n" + "=" * 50)
    print("最终测试报告 (与训练 Loss 结构对齐):")
    print(f"理论引导总损失 (TF Total)  : {metrics['tf_loss']:.5f}")
    print(f"真实自回归总损失 (AR Total) : {metrics['ar_loss']:.5f}")
    print("-" * 30)
    print(f"  ├─ 位置精确度 (Pos Loss)   : {metrics['ar_pos']:.5f}")
    print(f"  ├─ 动作流畅度 (Vel Loss)   : {metrics['ar_vel']:.5f}")
    print(f"  ├─ 动态相关性 (Corr/越高越好): {metrics['ar_corr']:.4f}")
    print(f"  ├─ 嘴形偏移量 (Bias Loss)  : {metrics['ar_bias']:.5f}")
    print(f"  └─ 对比/方差 (Contra/Var)  : {metrics['ar_contra']:.5f} / {metrics['ar_var']:.5f}")
    print("=" * 50)


if __name__ == "__main__":
    evaluate_model()