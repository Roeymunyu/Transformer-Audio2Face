import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import AudioFaceDataset
from model import FaceFormerDemoV3
from loss import FaceFormerLoss


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" 使用设备: {device}")

    DATA_DIR = "Dataset_Tensors"
    SEQ_LEN = 120
    BATCH_SIZE = 32
    EPOCHS = 60
    LR = 0.0001
    MOTION_DIM = 33

    train_dataset = AudioFaceDataset(DATA_DIR, seq_len=SEQ_LEN, stride=30, is_train=True)
    val_dataset = AudioFaceDataset(DATA_DIR, seq_len=SEQ_LEN, stride=30, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    model = FaceFormerDemoV3(motion_dim=MOTION_DIM).to(device)
    criterion = FaceFormerLoss(device=device, motion_dim=MOTION_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    best_val_loss = float('inf')

    print(f"\n 开始训练！总计 {EPOCHS} 轮...")
    for epoch in range(EPOCHS):

        # ==========================================
        # Teacher Forcing 调度（进一步压低到 0.1）
        #
        #   Epoch  1-3 :  TF = 1.0
        #   Epoch  4-12:  1.0 → 0.5
        #   Epoch 13-30:  0.5 → 0.1
        #   Epoch 31-60:  0.1 (30个epoch的深度AR训练)
        # ==========================================
        if epoch < 3:
            tf_ratio = 1.0
        elif epoch < 12:
            tf_ratio = 1.0 - 0.5 * (epoch - 3) / 9.0
        elif epoch < 30:
            tf_ratio = 0.5 - 0.4 * (epoch - 12) / 18.0
        else:
            tf_ratio = 0.1

        # 去噪强度
        noise_std = min(0.05, 0.01 + 0.04 * epoch / EPOCHS)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n--- Epoch [{epoch + 1:3d}/{EPOCHS}] | TF: {tf_ratio:.2%} | "
              f"Noise σ: {noise_std:.3f} | LR: {current_lr:.6f} ---")

        # ================= 训练 =================
        model.train()
        train_totals = {'loss': 0, 'pos': 0, 'vel': 0, 'contra': 0,
                        'var': 0, 'corr': 0, 'bias': 0}

        for audio_batch, face_batch in train_loader:
            audio_batch = audio_batch.to(device)
            face_batch = face_batch.to(device)
            optimizer.zero_grad()

            if tf_ratio >= 0.99:
                # 纯TF + 去噪
                noisy_face = face_batch + torch.randn_like(face_batch) * noise_std
                noisy_face = torch.clamp(noisy_face, 0.0, 1.0)
                predictions = model(audio_batch, noisy_face)
            else:
                # 混合TF + 去噪
                with torch.no_grad():
                    preds_tf = model(audio_batch, face_batch)

                b, s, d = face_batch.shape
                mask = torch.rand(b, s, 1, device=device) < tf_ratio
                mixed = torch.where(mask, face_batch, preds_tf.detach())
                mixed = mixed + torch.randn_like(mixed) * noise_std
                mixed = torch.clamp(mixed, 0.0, 1.0)

                predictions = model(audio_batch, mixed)

            total_loss, pos, vel, contra, var, corr, bias = criterion(predictions, face_batch)
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_totals['loss'] += total_loss.item()
            train_totals['pos'] += pos.item()
            train_totals['vel'] += vel.item()
            train_totals['contra'] += contra.item()
            train_totals['var'] += var.item()
            train_totals['corr'] += corr.item()
            train_totals['bias'] += bias.item()

        n_train = len(train_loader)

        # ================= 验证 =================
        model.eval()
        val_totals = {'loss': 0, 'pos': 0, 'vel': 0, 'contra': 0,
                      'var': 0, 'corr': 0, 'bias': 0}

        with torch.no_grad():
            for audio_batch, face_batch in val_loader:
                audio_batch = audio_batch.to(device)
                face_batch = face_batch.to(device)

                predictions = model(audio_batch, face_batch)
                total_loss, pos, vel, contra, var, corr, bias = criterion(predictions, face_batch)

                val_totals['loss'] += total_loss.item()
                val_totals['pos'] += pos.item()
                val_totals['vel'] += vel.item()
                val_totals['contra'] += contra.item()
                val_totals['var'] += var.item()
                val_totals['corr'] += corr.item()
                val_totals['bias'] += bias.item()

        n_val = len(val_loader)

        scheduler.step(val_totals['loss'] / n_val)

        # ================= 日志 =================
        print(f"  Train: Total={train_totals['loss']/n_train:.5f} | "
              f"Pos={train_totals['pos']/n_train:.5f} | "
              f"Vel={train_totals['vel']/n_train:.5f} | "
              f"Corr={train_totals['corr']/n_train:.4f} | "
              f"Bias={train_totals['bias']/n_train:.4f}")
        print(f"  Val:   Total={val_totals['loss']/n_val:.5f} | "
              f"Pos={val_totals['pos']/n_val:.5f} | "
              f"Vel={val_totals['vel']/n_val:.5f} | "
              f"Corr={val_totals['corr']/n_val:.4f} | "
              f"Bias={val_totals['bias']/n_val:.4f}")

        avg_val = val_totals['loss'] / n_val
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), "best_faceformer-v6.pth")
            print(f" 已保存 (Val: {best_val_loss:.5f})")

    print("\n 训练完毕！")


if __name__ == "__main__":
    train()