"""
control/loop.py — 主控制迴圈（在 daemon thread 內執行）。

實作 PRD「倒單擺與阻抗控制平滑切換系統」的核心狀態機：
    State 1  Swing-up   起擺 — 能量法將擺桿從下垂甩到頂端
    State 2  Balance    平衡 — LQR 穩定擺桿於正上方
    State 3  Impedance  阻抗控制 — 使用者按鈕觸發後接管旋臂扭矩輸出

依賴 SharedState 進行執行緒間通訊，不使用任何全域變數。
虛擬模式已移除，僅支援實體 QUBE-Servo 3（含擺桿附件）。
"""
import time
import threading
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np

from control.filters   import LowPassFilter
from control.impedance import ImpedanceDynamics
from control.safety    import SafetyChecker, VoltageStepLimiter
from control.step      import StepContext, run_step
from control.pendulum  import (
    PendulumMode, DerivativeEstimator, wrap_angle,
    swing_up_voltage, balance_voltage, is_near_top, is_switch_eligible,
)
from config import (
    FC_SPEED, FC_ACCEL, FC_FORCE, WARMUP_CYCLES,
    PENDULUM_VOLTAGE_LIMIT, PENDULUM_ENGAGE_ANGLE_DEG, PENDULUM_ENGAGE_RATE_RADS,
    IMPEDANCE_ENABLE_ANGLE_DEG, IMPEDANCE_ENABLE_RATE_RADS,
    IMPEDANCE_ENABLE_HOLD_CYCLES, BUMPLESS_TRANSFER_CYCLES,
)


@dataclass
class RoundConfig:
    """每輪實驗的不可變設定，用來減少 _run_one_round 的參數數量。"""
    rnd_idx:       int
    total_rounds:  int
    exp_time:      float
    dt:            float
    qube:          object
    has_current:   bool


def _try_import_qube():
    try:
        from pal.products.qube import QubeServo3
        return QubeServo3
    except ImportError:
        return None


def control_loop(params: dict, state,
                 log_cb:            Callable[[str], None],
                 status_cb:         Callable[[bool, str, str], None],
                 round_done_cb:     Callable[[list, int], None],
                 all_done_cb:       Callable[[], None],
                 new_round_cb:      Callable[[int, threading.Event], None],
                 safety_alert_cb:   Callable[[str], None],
                 pendulum_state_cb: Optional[Callable[[str, bool], None]] = None):
    """
    主控制迴圈。

    Args:
        params:         {"sample_time", "exp_time", "total_rounds"}
        state:          SharedState 實例
        log_cb:         寫日誌的 callback
        status_cb:      更新連線狀態的 callback (connected, device, error)
        round_done_cb:  每輪完成後的 callback (round_data, round_number)
        all_done_cb:    全部輪次完成後的 callback
        new_round_cb:   新輪開始前的 callback (round_number, clear_event)
        safety_alert_cb:   觸發緊急停止的 callback (reason)
        pendulum_state_cb: 狀態機切換時的 callback (mode_name, switch_enabled)
    """
    dt           = params["sample_time"]
    total_rounds = params["total_rounds"]
    exp_time     = params["exp_time"]

    if pendulum_state_cb is None:
        pendulum_state_cb = lambda mode_name, enabled: None

    safety  = SafetyChecker(log_cb, state.emergency)
    imp_dyn = ImpedanceDynamics()

    QubeServo3 = _try_import_qube()
    if QubeServo3 is None:
        msg = "找不到 pal 套件，請確認已安裝 QUBE 驅動。"
        log_cb(f"[ERROR] {msg}")
        status_cb(False, "None", msg)
        all_done_cb(); return

    log_cb(f"[INFO] 實體模式 | dt={dt}s  exp={exp_time}s  "
           f"rounds={total_rounds}  warmup={WARMUP_CYCLES}cycles")

    try:
        # pendulum=1：本系統需要擺桿附件才能執行起擺 / 平衡 / 阻抗切換狀態機
        device_ctx = QubeServo3(hardware=1, pendulum=1, readMode=0)
    except Exception as e:
        msg = f"實體連線失敗：{e}"
        log_cb(f"[ERROR] {msg}")
        status_cb(False, "None", msg)
        all_done_cb(); return

    status_cb(True, "QubeServo3", "")

    with device_ctx as qube:
        _has_current = hasattr(qube, "motorCurrent")
        if not _has_current:
            log_cb("[WARN] 無 motorCurrent 屬性，電流固定為 0.0A，外力估算停用")

        for rnd_idx in range(total_rounds):
            if state.kill.is_set() or state.emergency.is_set():
                break

            cfg = RoundConfig(rnd_idx, total_rounds, exp_time, dt,
                              qube, _has_current)
            _run_one_round(cfg, safety, imp_dyn, state,
                           log_cb, round_done_cb, new_round_cb,
                           safety_alert_cb, pendulum_state_cb)

    status_cb(False, "None", "")
    log_cb("[INFO] 控制迴圈結束")
    all_done_cb()


def _run_one_round(cfg: RoundConfig,
                   safety, imp_dyn, state,
                   log_cb:            Callable[[str], None],
                   round_done_cb:     Callable[[list, int], None],
                   new_round_cb:      Callable[[int, threading.Event], None],
                   safety_alert_cb:   Callable[[str], None],
                   pendulum_state_cb: Callable[[str, bool], None]):
    """執行一輪實驗：Swing-up → Balance → (使用者觸發) Impedance 狀態機。"""
    safety.reset()
    imp_dyn.reset()

    clear_event = threading.Event()
    new_round_cb(cfg.rnd_idx + 1, clear_event)
    clear_event.wait(timeout=2.0)

    imp_ctx = StepContext(
        LowPassFilter(FC_SPEED, cfg.dt),
        LowPassFilter(FC_ACCEL, cfg.dt),
        LowPassFilter(FC_FORCE, cfg.dt),
        imp_dyn,
    )
    deriv        = DerivativeEstimator(cfg.dt)
    step_limiter = VoltageStepLimiter()

    with state.data_lock:
        state.round_history.clear()

    log_cb(f"[ROUND {cfg.rnd_idx + 1}/{cfg.total_rounds}] 開始 — 起擺中")

    mode              = PendulumMode.SWINGUP
    stable_count      = 0
    switch_enabled    = False
    last_bal_voltage  = 0.0
    blend_remaining   = 0
    pendulum_state_cb(mode.value, switch_enabled)

    start_time = time.time()
    timestamp  = 0.0

    while (timestamp < cfg.exp_time
           and not state.kill.is_set()
           and not state.emergency.is_set()):

        while state.pause.is_set() and not state.kill.is_set():
            time.sleep(0.05)

        t0 = time.time()

        try:
            cfg.qube.read_outputs()
            theta   = float(np.asarray(cfg.qube.motorPosition).flat[0])
            alpha_f = float(np.asarray(cfg.qube.pendulumPosition).flat[0])
            alpha   = wrap_angle(alpha_f)
            current = (float(np.asarray(cfg.qube.motorCurrent).flat[0])
                       if cfg.has_current else 0.0)

            theta_dot, alpha_dot = deriv.update(theta, alpha_f)
            ctrl_params = state.get_params()
            K, B, M, Kp, Kd, theta_d = ctrl_params

            # 手動重置：無論目前處於哪個狀態，一律返回起擺（PRD 3.2 狀態鎖定的唯一出口）
            if state.mode_reset.is_set():
                state.mode_reset.clear()
                mode = PendulumMode.SWINGUP
                stable_count = 0
                switch_enabled = False
                deriv.reset()
                log_cb("[STATE] 手動重置 → 起擺")
                pendulum_state_cb(mode.value, switch_enabled)

            if mode in (PendulumMode.SWINGUP, PendulumMode.BALANCE):
                safe, reason = safety.check(theta, theta_dot, current, 0.0)
                if not safe:
                    cfg.qube.write_voltage(0.0)
                    safety_alert_cb(reason)
                    return

                active_mode = mode

                if mode is PendulumMode.SWINGUP:
                    voltage = swing_up_voltage(alpha, alpha_dot)

                    # 角度 + 角速度須同時達標才「接住」擺桿並切入平衡控制：起擺過程中
                    # 擺桿每次擺盪都會高速路過頂端，若只看角度會誤判為已平衡（詳見
                    # config.py::PENDULUM_ENGAGE_RATE_RADS 的說明）。
                    if is_switch_eligible(alpha, alpha_dot,
                                          PENDULUM_ENGAGE_ANGLE_DEG,
                                          PENDULUM_ENGAGE_RATE_RADS):
                        mode = PendulumMode.BALANCE
                        deriv.reset()
                        stable_count = 0
                        log_cb("[STATE] 起擺 → 平衡")
                        pendulum_state_cb(mode.value, switch_enabled)

                else:  # BALANCE
                    # 平衡律使用旋轉臂角度的翻轉慣例（沿用已於硬體驗證的 BALANCE_K）
                    voltage = balance_voltage(-theta, alpha, -theta_dot, alpha_dot,
                                              PENDULUM_VOLTAGE_LIMIT)
                    last_bal_voltage = voltage

                    if not is_near_top(alpha, PENDULUM_ENGAGE_ANGLE_DEG):
                        # 邊界條件（PRD 7.1）：致能後尚未按下按鈕但擺桿掉落 → 重新起擺
                        mode = PendulumMode.SWINGUP
                        stable_count = 0
                        if switch_enabled:
                            switch_enabled = False
                        deriv.reset()
                        log_cb("[STATE] 擺桿掉落 → 重新起擺")
                        pendulum_state_cb(mode.value, switch_enabled)
                    else:
                        eligible = is_switch_eligible(
                            alpha, alpha_dot,
                            IMPEDANCE_ENABLE_ANGLE_DEG, IMPEDANCE_ENABLE_RATE_RADS)
                        stable_count = stable_count + 1 if eligible else 0
                        new_enabled = stable_count >= IMPEDANCE_ENABLE_HOLD_CYCLES
                        if new_enabled != switch_enabled:
                            switch_enabled = new_enabled
                            pendulum_state_cb(mode.value, switch_enabled)

                        if switch_enabled and state.impedance_switch.is_set():
                            state.impedance_switch.clear()
                            # 記錄切換瞬間的旋轉臂位置作為阻抗控制的平衡點（PRD 3.3）
                            state.set_params(K, B, M, Kp, Kd, theta)
                            imp_dyn.reset()
                            imp_ctx.prev_theta = None
                            imp_ctx.prev_omega = 0.0
                            imp_ctx.spd_f.reset()
                            imp_ctx.acc_f.reset()
                            imp_ctx.frc_f.reset()
                            mode = PendulumMode.IMPEDANCE
                            switch_enabled = False
                            stable_count = 0
                            blend_remaining  = BUMPLESS_TRANSFER_CYCLES
                            log_cb(f"[STATE] 平衡 → 阻抗控制  平衡點 theta_d={theta:.4f} rad")
                            pendulum_state_cb(mode.value, switch_enabled)

                row = {
                    "round": cfg.rnd_idx + 1, "time": round(timestamp, 4),
                    "mode": active_mode.value,
                    "theta_rad": round(theta, 6), "theta_d_rad": 0.0,
                    "theta_cmd_rad": round(theta, 6), "omega_rads": round(theta_dot, 6),
                    "alpha_rad": round(alpha, 6), "alpha_dot_rads": round(alpha_dot, 6),
                    "voltage_V": round(voltage, 6), "current_A": round(current, 6),
                    "force_est_Nm": 0.0,
                    "K_Nm_rad": round(K, 3), "B_Nms_rad": round(B, 4),
                    "M_kgm2": round(M, 4), "Kp": round(Kp, 3), "Kd": round(Kd, 3),
                }
                theta_d_row, theta_cmd_row, omega_row, force_row = (
                    0.0, theta, theta_dot, 0.0)

            else:  # IMPEDANCE — PRD 3.3/7.2：不再對擺桿角度給予任何補償，狀態鎖定忽略掉落
                safe, reason, imp_voltage, row = run_step(
                    theta, current, imp_ctx, ctrl_params,
                    safety, cfg.dt, cfg.rnd_idx + 1, timestamp,
                    alpha=alpha, alpha_dot=alpha_dot, mode="impedance")

                if not safe:
                    cfg.qube.write_voltage(0.0)
                    safety_alert_cb(reason)
                    return

                if blend_remaining > 0:
                    # Bumpless transfer：由平衡控制輸出線性過渡到阻抗控制輸出
                    frac    = 1.0 - blend_remaining / BUMPLESS_TRANSFER_CYCLES
                    voltage = last_bal_voltage * (1.0 - frac) + imp_voltage * frac
                    blend_remaining -= 1
                else:
                    voltage = imp_voltage
                row["voltage_V"] = round(voltage, 6)
                theta_d_row   = row["theta_d_rad"]
                theta_cmd_row = row["theta_cmd_rad"]
                omega_row     = row["omega_rads"]
                force_row     = row["force_est_Nm"]

            # 統一的電壓步階飽和上限：吸收掉落衝擊或任何模式切換瞬間的扭矩突變（PRD 6.1）
            voltage = step_limiter.clamp(voltage)
            row["voltage_V"] = round(voltage, 6)

            cfg.qube.write_voltage(voltage)

            with state.data_lock:
                state.push_buffers(timestamp, theta, theta_d_row,
                                   theta_cmd_row, omega_row, voltage, force_row)
                state.round_history.append(row)

        except Exception as e:
            log_cb(f"[ERROR] {type(e).__name__}: {e}")
            try:
                cfg.qube.write_voltage(0.0)
            except Exception:
                pass
            safety_alert_cb(f"迴圈例外：{type(e).__name__}: {e}")
            return

        elapsed   = time.time() - t0
        time.sleep(max(0.0, cfg.dt - elapsed))
        timestamp = time.time() - start_time

    cfg.qube.write_voltage(0.0)

    if not state.kill.is_set() and not state.emergency.is_set():
        with state.data_lock:
            state.round_counter += 1
            state.all_rounds_history.append(list(state.round_history))
            rnd = state.round_counter

        log_cb(f"[ROUND {rnd}] 完成")
        round_done_cb(list(state.round_history), rnd)
