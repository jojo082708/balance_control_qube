"""
config.py — 所有常數集中在此，修改參數只需改這一個檔案。
"""
import math
import numpy as np

# ── 馬達參數（系統辨識後在此修改）────────────────────────────────────────────
Kt         = 0.042      # 力矩常數   [N·m/A]
J_motor    = 4.0e-6    # 轉子慣量   [kg·m²]
B_friction = 1.0e-5    # 黏性摩擦   [N·m·s/rad]

# ── 安全限制 ──────────────────────────────────────────────────────────────────
ANGLE_LIMIT_RAD  = math.radians(270)   # ±270°
SPEED_LIMIT_RADS = 50.0                # [rad/s]
VOLTAGE_LIMIT    = 10.0                # [V]
CURRENT_LIMIT    = 4.0                 # [A]
FORCE_EST_LIMIT  = 1.0                 # [N·m]
WARMUP_CYCLES    = 50                  # 暖機週期數（dt=0.002s → 100 ms）

# ── 濾波器截止頻率 [Hz] ────────────────────────────────────────────────────────
FC_SPEED = 40.0
FC_ACCEL = 15.0
FC_FORCE = 15.0

# ── 資料緩衝 ──────────────────────────────────────────────────────────────────
BUFFER_SIZE = 4000

# 固定 CSV / Excel 欄位順序（不隨參數值改變）
ROW_FIELDS = [
    "round", "time", "mode",
    "theta_rad", "theta_d_rad", "theta_cmd_rad",
    "omega_rads", "alpha_rad", "alpha_dot_rads",
    "voltage_V", "current_A", "force_est_Nm",
    "K_Nm_rad", "B_Nms_rad", "M_kgm2", "Kp", "Kd",
]

# ── 倒單擺參數（Furuta pendulum — QUBE-Servo 3 擺桿）────────────────────────
# 系統辨識後請依實際硬體修改；alpha=0 定義為擺桿正上方（平衡點），
# alpha=±π 為自然下垂位置（與 pal QubeServo3.pendulumPosition 的 wrap 慣例一致）。
PENDULUM_MASS       = 0.024                                  # Mp  擺桿質量 [kg]
PENDULUM_LENGTH     = 0.129                                  # Lp  擺桿長度 [m]
PENDULUM_COM_RADIUS = PENDULUM_LENGTH / 2                     # lp  轉軸到質心距離 [m]
PENDULUM_INERTIA    = (1.0 / 3.0) * PENDULUM_MASS * PENDULUM_LENGTH ** 2  # Jp [kg·m²]（均勻桿近似）
GRAVITY              = 9.81                                   # [m/s²]

# LQR 平衡增益（狀態順序：theta, alpha, theta_dot, alpha_dot）
# 已於 QUBE-Servo 3 實體硬體驗證（沿用原 balance_control_qube.py 之增益）
BALANCE_K = np.array([-1.2247, 24.9044, -0.6877, 3.1321])

# 起擺（能量法, energy-based swing-up — Åström & Furuta 1996, Eq. 8）
SWINGUP_GAIN            = 40.0   # μ  能量誤差 → 電壓增益（對應文獻中的 k）
SWINGUP_VOLTAGE_LIMIT   = 3.0    # [V] 起擺期間電壓飽和上限（低於平衡/阻抗上限，降低對機構的衝擊）
# 理論正確值為 -1.0（見 control/pendulum.py::swing_up_voltage 的推導）。
# 僅當實體硬體編碼器 / 馬達接線極性相反、導致起擺方向錯誤時才改為 +1.0。
SWINGUP_DIRECTION_SIGN  = -1.0

# 倒單擺微分濾波器截止頻率 [Hz]
FC_THETA_DOT = 50.0
FC_ALPHA_DOT = 100.0

# ── 狀態機切換判定（PRD 3.2 / 5.1）────────────────────────────────────────────
PENDULUM_ENGAGE_ANGLE_DEG    = 10.0   # 起擺 → 平衡 的進入角度（也是平衡 → 起擺的掉落角度）
IMPEDANCE_ENABLE_ANGLE_DEG   = 5.0    # 「切換至阻抗控制」按鈕致能角度容許範圍
IMPEDANCE_ENABLE_RATE_RADS   = 0.5    # 按鈕致能角速度容許範圍 [rad/s]
IMPEDANCE_ENABLE_HOLD_CYCLES = 100    # 需連續 N 個取樣週期穩定才致能按鈕

# 平滑切換（bumpless transfer）：切至阻抗控制瞬間，電壓從平衡控制輸出線性過渡到阻抗控制輸出
BUMPLESS_TRANSFER_CYCLES = 50

# 倒單擺專屬安全飽和上限（control/safety.py 使用）
PENDULUM_VOLTAGE_LIMIT     = 8.0   # [V] 起擺/平衡期間電壓飽和上限
MAX_VOLTAGE_STEP_PER_CYCLE = 1.0   # [V] 每週期允許的最大電壓變化量（掉落衝擊 / 模式切換瞬間的保護）

# ── GUI 更新速率 ───────────────────────────────────────────────────────────────
PLOT_DOWNSAMPLE_PTS = 800   # 繪圖最大取樣點數
POLL_INTERVAL_MS    = 200   # 控制 thread 輪詢間隔 [ms]
PLOT_INTERVAL_MS    = 80    # 繪圖更新間隔 [ms]
THREAD_WAIT_MAX     = 30    # 最多等待舊 thread 幾次（× POLL_INTERVAL_MS）
