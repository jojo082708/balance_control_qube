"""
control/safety.py — 安全限制檢查器。

每個控制週期呼叫一次 check()。
- 角度 / 速度 / 電流超限 → 觸發緊急停止（設定 emergency event）
- 外力估算異常         → 只警告 + 本週期外力歸零，不停止
- 暖機期               → 跳過電流 / 外力檢查

VoltageStepLimiter 另外提供倒單擺掉落衝擊 / 模式切換瞬間的扭矩飽和限制（PRD 6.1）。
"""
import math
from config import (ANGLE_LIMIT_RAD, SPEED_LIMIT_RADS,
                    CURRENT_LIMIT, FORCE_EST_LIMIT, WARMUP_CYCLES,
                    MAX_VOLTAGE_STEP_PER_CYCLE)


class SafetyChecker:

    def __init__(self, log_cb, emergency_event):
        self._log       = log_cb
        self._emergency = emergency_event
        self._triggered = False
        self._cycle     = 0

    def check(self, theta: float, omega: float,
              current: float, force_est: float) -> tuple:
        """回傳 (safe: bool, reason: str)。"""
        if self._triggered:
            return False, "緊急停止已觸發"

        self._cycle += 1
        in_warmup = self._cycle <= WARMUP_CYCLES

        if abs(theta) > ANGLE_LIMIT_RAD:
            return self._trigger(
                f"角度超限 {math.degrees(theta):.1f}° "
                f"(限制 ±{math.degrees(ANGLE_LIMIT_RAD):.0f}°)")

        if abs(omega) > SPEED_LIMIT_RADS:
            return self._trigger(
                f"速度超限 {omega:.2f} rad/s "
                f"(限制 ±{SPEED_LIMIT_RADS} rad/s)")

        if not in_warmup and abs(current) > CURRENT_LIMIT:
            return self._trigger(
                f"電流異常 {current:.3f} A (限制 {CURRENT_LIMIT} A)")

        if not in_warmup and abs(force_est) > FORCE_EST_LIMIT:
            self._log(
                f"[WARN] 外力估算異常 {force_est:.4f} N·m "
                f"(限制 {FORCE_EST_LIMIT} N·m)，本週期歸零")
            return True, "force_zero"

        return True, ""

    def _trigger(self, msg: str) -> tuple:
        self._triggered = True
        self._emergency.set()
        self._log(f"[EMERGENCY] {msg}")
        return False, msg

    def reset(self):
        self._triggered = False
        self._cycle     = 0
        self._emergency.clear()


def saturate(voltage: float, limit: float) -> float:
    """簡單電壓飽和上限（PRD 6.1：倒單擺掉落衝擊 / 切換瞬間扭矩上限）。"""
    return max(-limit, min(limit, voltage))


class VoltageStepLimiter:
    """
    限制電壓每週期的最大變化量，避免倒單擺掉落的衝擊或狀態切換瞬間的扭矩
    突變對機械結構造成損壞（PRD 6.1）。

    在硬體控制迴圈（500Hz–1000Hz）的單一週期內完成計算，不含任何阻塞操作。
    """

    def __init__(self, max_step: float = MAX_VOLTAGE_STEP_PER_CYCLE):
        self._max_step = max_step
        self._last      = 0.0

    def reset(self, value: float = 0.0):
        self._last = value

    def clamp(self, voltage: float) -> float:
        delta = voltage - self._last
        if delta > self._max_step:
            voltage = self._last + self._max_step
        elif delta < -self._max_step:
            voltage = self._last - self._max_step
        self._last = voltage
        return voltage
