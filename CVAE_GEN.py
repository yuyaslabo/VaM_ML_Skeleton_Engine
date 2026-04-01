import torch
import torch.nn as nn
import socket
import time
import os
import glob
import numpy as np
import tkinter as tk
from tkinter import ttk
import threading
from collections import deque

# --- 設定 ---
TARGET_BONES = [
    "hipControl","chestControl", "headControl",
        "rHandControl","lHandControl","rFootControl","lFootControl",
		"rKneeControl", "lKneeControl"
]
TARGET_IP = ""
TARGET_PORT = 0

cond_names = []
    
flag_path = "flag.csv"
if os.path.exists(flag_path):
    with open(flag_path, "r", encoding="utf-8") as f:
        cond_names = f.read().splitlines()
    print(f"flag.csv loaded: {cond_names}")
else:
    print(f"error: {flag_path} not found")
    sys.exit(1)
    
target_ip_path = "target_ip.csv"
if os.path.exists(target_ip_path):
    with open(target_ip_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
        if lines:
            # 最初の1行をコロンで分割
            parts = lines[0].strip().split(':')
            TARGET_IP = parts[0]
            
            # コロンの後に値があればポートを更新
            if len(parts) > 1:
                try:
                    TARGET_PORT = int(parts[1])
                except ValueError:
                    print(f"Warning: The port number is incorrect. Use the default {TARGET_PORT}.")
        else:
            print(f"Error: {target_ip_path} is empty.")
            sys.exit(1)
    print(f"Loading successful: IP={TARGET_IP}, PORT={TARGET_PORT}")
else:
    print(f"Error: {target_ip_path} not found.")
    sys.exit(1)

INPUT_DIM = len(TARGET_BONES) * 9 
COND_DIM = len(cond_names)
LATENT_DIM = 24
HIDDEN_DIM = 1024
SEQ_LEN = 20

MODEL_PATH = "pose_gru_cvae.pth"
LEAP_COND_SPEED = 0.05
dt_factor = 0.04  # 初期値 (元コードの dt = 0.1 に相当)

def update_lerp():
    #常に現在の値を目標値に近づけるループ
    global cond_values, target_cond
    while True:
        # 線形補間: 現在値 = 現在値 + (目標値 - 現在値) * 速度
        cond_values += (target_cond - cond_values) * LEAP_COND_SPEED
        time.sleep(0.04) # 約30FPSで更新

def get_latest_checkpoint(dir_path):
    # 指定ディレクトリ内の .pth ファイルをすべて取得
    list_of_files = glob.glob(os.path.join(dir_path, '*.pth'))
    if not list_of_files:
        return None
    # 更新日時(mtime)が最大（最新）のファイルパスを返す
    return max(list_of_files, key=os.path.getmtime)
# --- モデル定義 ---
class PoseGRU_CVAE(nn.Module):
    def __init__(self, input_dim, cond_dim, latent_dim, hidden_dim):
        super().__init__()
        self.enc_gru = nn.GRU(input_dim + cond_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.dec_gru = nn.GRU(latent_dim + cond_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def decode(self, z, c_seq):
        # z: [1, latent], c_seq: [1, seq, cond]
        z_ext = z.unsqueeze(1).repeat(1, SEQ_LEN, 1)
        dec_in = torch.cat([z_ext, c_seq], dim=-1)
        out, _ = self.dec_gru(dec_in)
        return self.fc_out(out)


def rotation_6d_to_quaternion(d6):
    x_raw, y_raw = d6[:, 0:3], d6[:, 3:6]
    x = x_raw / torch.norm(x_raw, dim=1, keepdim=True).clamp(min=1e-6)
    z = torch.cross(x, y_raw, dim=1)
    z = z / torch.norm(z, dim=1, keepdim=True).clamp(min=1e-6)
    y = torch.cross(z, x, dim=1)
    matrix = torch.stack([x, y, z], dim=2)
    m00, m11, m22 = matrix[:, 0, 0], matrix[:, 1, 1], matrix[:, 2, 2]
    q_w = torch.sqrt((1.0 + m00 + m11 + m22).clamp(min=1e-6)) * 0.5
    q_x = (matrix[:, 2, 1] - matrix[:, 1, 2]) / (4 * q_w + 1e-6)
    q_y = (matrix[:, 0, 2] - matrix[:, 2, 0]) / (4 * q_w + 1e-6)
    q_z = (matrix[:, 1, 0] - matrix[:, 0, 1]) / (4 * q_w + 1e-6)
    return torch.stack([q_x, q_y, q_z, q_w], dim=1)

# --- グローバル変数 ---
cond_values = np.zeros(COND_DIM, dtype=np.float32)
target_cond = np.zeros(COND_DIM, dtype=np.float32)
cond_values[0] = 1.0
target_cond[0] = 1.0

running = True

MODEL_DIR = "./"  # モデルが保存されているフォルダ
latest_path = get_latest_checkpoint(MODEL_DIR)

model = PoseGRU_CVAE(INPUT_DIM, COND_DIM, LATENT_DIM, HIDDEN_DIM)

if latest_path:
    print(f"Loading latest model: {latest_path}")
    model.load_state_dict(torch.load(latest_path, map_location='cpu'))
else:
    print("No .pth files found.")

# COND LEAPスレッド
threading.Thread(target=update_lerp, daemon=True).start()
# --- 推論ループ (10Hz) ---
def inference_thread():
    global dt_factor
    
    # head補間用履歴
    prev_quat = {
        "headControl": np.array([0.0, 0.0, 0.0, 1.0]),
    }
    # 潜在空間の初期状態
    wander_z = torch.zeros(1, LATENT_DIM)
    velocity_z = torch.zeros(1, LATENT_DIM)
    # 履歴管理
    pose_history = deque([torch.zeros(INPUT_DIM) for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)
    
    # 前回の確定ポーズを保持する変数 (最初は Baseline または zeros)
    last_sent_pose = torch.zeros(INPUT_DIM)
    # ポーズの追従速度 (0.0 ～ 1.0)
    POSE_LERP_FACTOR = 0.29
    
    frame_count = 0  # 送信回数をカウント
    WARMUP_FRAMES = 20  # 10Hzなので10フレームで約1秒
    
    #theta = 0.04
    theta = 0.04
    mu_z = torch.zeros(1, LATENT_DIM)
    #sigma_z = 0.3  # 揺れをしっかり出す
    sigma_z = 0.3
    
    # LPF用の変数
    z_smooth = torch.zeros(1, LATENT_DIM)
    Z_LERP_FACTOR = 0.3 # 小さいほどヌルヌル、大きいほど機敏
    # --- クォータニオンSLERP ---
    def slerp_quaternion(q1, q2, t):
        """
        q1, q2: np.array([x, y, z, w])
        t: 0.0 ~ 1.0
        """
        dot = np.dot(q1, q2)

        # 反転で最短経路にする
        if dot < 0.0:
            q2 = -q2
            dot = -dot

        # ほぼ同一なら線形補間にフォールバック
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)

        theta_0 = np.arccos(dot)
        theta = theta_0 * t

        sin_theta = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)

        s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s2 = sin_theta / sin_theta_0

        return (s1 * q1) + (s2 * q2)
        
    def send_pose(final_pose):
        nonlocal frame_count

        # --- 固定設定 ---
        X_FACTOR = 0.3
        ROT_STRENGTH = 0.7
        SLERP_T = 0.2  # 補間量

        frame_count += 1
        if frame_count <= WARMUP_FRAMES:
            if frame_count == 1:
                print("Warming up model context...")
            if frame_count == WARMUP_FRAMES:
                print("Warming up Done!")
            return

        reshaped = final_pose.view(len(TARGET_BONES), 9)
        pos_np = reshaped[:, :3].numpy()
        quat_np = rotation_6d_to_quaternion(reshaped[:, 3:]).numpy()

        for i, bone in enumerate(TARGET_BONES):

            if bone == "headControl":
                identity = np.array([0.0, 0.0, 0.0, 1.0])

                # 強さ調整（元ロジック維持）
                target_q = identity + (quat_np[i] - identity) * ROT_STRENGTH
                target_q /= np.linalg.norm(target_q)

                quat_np[i] = slerp_quaternion(prev_quat["headControl"], target_q, SLERP_T)

                prev_quat["headControl"] = quat_np[i].copy()

                pos_np[i, 0] *= X_FACTOR
                #pos_np[i, 2] *= X_FACTOR

            elif bone == "chestControl":
                pos_np[i, 0] *= X_FACTOR
                #pos_np[i, 2] *= X_FACTOR

            elif bone == "hipControl":
                pos_np[i, 0] *= X_FACTOR
                #pos_np[i, 2] *= X_FACTOR

        payload = "|".join(
            f"{bone},{p[0]:.5f},{p[1]:.5f},{p[2]:.5f},"
            f"{q[0]:.5f},{q[1]:.5f},{q[2]:.5f},{q[3]:.5f}"
            for bone, p, q in zip(TARGET_BONES, pos_np, quat_np)
        )

        sock.sendto(payload.encode(), (TARGET_IP, TARGET_PORT)) 
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    
    
    
    
    

    print("Inference loop with Autonomous Wander started...")
    try:
        while running:
            loop_start = time.time()
            
            # --- 2. 内部でさらに細かく物理演算を回す (擬似的な高頻度化) ---
            # 10Hzの1フレームの間に、5回分物理を回して「うねり」を滑らかにする
            for _ in range(5):
                drift = theta * (mu_z - wander_z)
                diffusion = sigma_z * torch.randn(1, LATENT_DIM)
                accel = drift + diffusion
                
                
                dt = dt_factor  # 通常のスピード
                
                velocity_z = velocity_z * 0.9 + accel * dt
                wander_z += velocity_z

            # --- 3. Zのスムージング (カクつき除去の決定打) ---
            # モデルに渡す直前に、前回の z と今回の z を滑らかにつなぐ
            z_smooth = z_smooth + (wander_z - z_smooth) * Z_LERP_FACTOR
            
            # --- 4. 履歴と条件の準備 ---
            history_x = torch.stack(list(pose_history)).unsqueeze(0).float()
            c_vec = torch.from_numpy(np.array([cond_values])).float()
            c_vec = c_vec / (torch.sum(c_vec) + 1e-6)
            c_seq = c_vec.unsqueeze(1).repeat(1, SEQ_LEN, 1)

            with torch.no_grad():
                # Encoderからの文脈に、ヌルヌル動く z_smooth を足す
                enc_in = torch.cat([history_x, c_seq], dim=-1)
                _, h = model.enc_gru(enc_in)
                z_mu = model.fc_mu(h[-1])
                
                z_final = z_mu + z_smooth  # カクつきのないZ
                
                generated_seq = model.decode(z_final, c_seq)
                current_pose = generated_seq[0, -1, :]
                
                
            
                
            # --- ここで Lerp (線形補間) をかける ---
            # formula: last + (target - last) * factor
            final_pose = last_sent_pose + (current_pose - last_sent_pose) * POSE_LERP_FACTOR
            last_sent_pose = final_pose.clone()
            # 3. 履歴の更新
            pose_history.append(final_pose)

            send_pose(final_pose)
                    
            time.sleep(max(0, 1/10 - (time.time() - loop_start)))
    except Exception as e:
        print(f"Error in inference: {e}")

# --- GUI構築 (スライダー4本) ---
root = tk.Tk()
root.title("VaM Skeleton Engine")
root.geometry("400x950") 
root.attributes("-topmost", True)

style = ttk.Style()
style.configure("Normal.TLabel", foreground="gray50")
style.configure("Highlight.TLabel", foreground="black")
target_indices = [0, 4, 11, 13, 15, 16, 20, 21, 22]

# 初期値リスト（要素数に合わせて作成）
target_cond = [1.0 if i == 0 else 0.0 for i in range(len(cond_names))]

sliders, labels = [], []

# --- dt操作用スライダーの追加 ---
def on_dt_move(val):
    global dt_factor
    dt_factor = float(val)
    dt_val_label.config(text=f"{dt_factor:.3f}")

dt_frame = ttk.LabelFrame(root, text="Movement Speed (dt)")
dt_frame.pack(fill="x", padx=20, pady=10)

dt_val_label = ttk.Label(dt_frame, text=f"{dt_factor:.3f}", width=6)
dt_val_label.pack(side="right")

# from_=0.01 (0だと動かなくなるため)
dt_slider = ttk.Scale(dt_frame, from_=0.001, to=0.3, orient="horizontal", command=on_dt_move)
dt_slider.set(dt_factor)
dt_slider.pack(side="left", fill="x", expand=True, padx=10)

def on_cond_move(idx, val):
    target_cond[idx] = float(val)
    labels[idx].config(text=f"{float(val):.2f}")

# --- ここからスライダーの生成（関数の外に出す！） ---
for i, name in enumerate(cond_names):
    frame = ttk.Frame(root)
    frame.pack(fill="x", padx=20, pady=5)
    
    current_style = "Highlight.TLabel" if i in target_indices else "Normal.TLabel"
    
    ttk.Label(frame, text=name, width=10, style=current_style).pack(side="left")
    
    val_label = ttk.Label(frame, text=f"{target_cond[i]:.2f}", width=6, style=current_style)
    val_label.pack(side="right")
    labels.append(val_label)
    
    s = ttk.Scale(frame, from_=0.0, to=1.0, orient="horizontal", 
                  command=lambda v, idx=i: on_cond_move(idx, v))
    s.set(target_cond[i])
    s.pack(side="left", fill="x", expand=True, padx=10)
    sliders.append(s)

def reset_all():
    for i, s in enumerate(sliders):
        val = 1.0 if i == 0 else 0.0
        s.set(val)
        labels[i].config(text=f"{val:.2f}")
        target_cond[i] = val

#ttk.Button(root, text="Reset to IDLE", command=reset_all).pack(pady=20)

# スレッド開始
threading.Thread(target=inference_thread, daemon=True).start()

def on_close():
    global running
    running = False
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()