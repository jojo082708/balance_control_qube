"""
control/pendulum.py — 倒單擺起擺（swing-up）與 LQR 平衡控制。

角度慣例：alpha = 0 為擺桿正上方（平衡點），alpha = ±π 為自然下垂位置，
與 pal QubeServo3.pendulumPosition 經過 wrap_angle() 後的慣例一致。

三個運作狀態（PRD 3.1）由 PendulumMode 定義，實際的狀態機邏輯在 control/loop.py。
本模組只提供純函式 / 無副作用的控制律計算，方便單元測試。
"""
import enum
import math
import numpy as np

from control.filters import LowPassFilter
from config import (
    BALANCE_K, SWINGUP_GAIN, SWINGUP_VOLTAGE_LIMIT, SWINGUP_DIRECTION_SIGN,
    PENDULUM_MASS, PENDULUM_COM_RADIUS, PENDULUM_INERTIA, GRAVITY,
    FC_THETA_DOT, FC_ALPHA_DOT,
)


class PendulumMode(enum.Enum):
    SWINGUP   = "swingup"
    BALANCE   = "balance"
    IMPEDANCE = "impedance"


def wrap_angle(angle_f: float) -> float:
    """將連續（未包裹）角度包裹到 (-π, π]，0 為擺桿正上方。"""
    return float(np.mod(angle_f + math.pi, 2 * math.pi) - math.pi)


class DerivativeEstimator:
    """
    對 theta / alpha 做微分估算，並以低通濾波器平滑。

    alpha 一律微分「未包裹」的原始讀值（alpha_f），避免 alpha 在 ±π 邊界
    wrap 造成的微分尖峰（沿用原 balance_control_qube.py 的作法）。
    """

    def __init__(self, dt: float):
        self._dt            = dt
        self._theta_filter   = LowPassFilter(FC_THETA_DOT, dt)
        self._alpha_filter   = LowPassFilter(FC_ALPHA_DOT, dt)
        self._prev_theta     = None
        self._prev_alpha_f   = None

    def reset(self):
        """Bumpless transfer：狀態切換瞬間清空微分歷史，避免大幅擺動殘留造成電壓尖峰。"""
        self._theta_filter.reset()
        self._alpha_filter.reset()
        self._prev_theta   = None
        self._prev_alpha_f = None

    def update(self, theta: float, alpha_f: float) -> tuple:
        """回傳 (theta_dot, alpha_dot)，皆已濾波。"""
        if self._prev_theta is None:
            self._prev_theta = theta
        if self._prev_alpha_f is None:
            self._prev_alpha_f = alpha_f

        theta_dot_raw = (theta - self._prev_theta) / self._dt
        alpha_dot_raw = (alpha_f - self._prev_alpha_f) / self._dt
        self._prev_theta   = theta
        self._prev_alpha_f = alpha_f

        return (self._theta_filter.update(theta_dot_raw),
                self._alpha_filter.update(alpha_dot_raw))


def pendulum_energy(alpha: float, alpha_dot: float) -> float:
    """
    E = 0.5·Jp·alpha_dot² + Mp·g·lp·cos(alpha)，於 alpha=0（正上方）最大。

    與 Åström & Furuta (1996, "Swinging Up a Pendulum by Energy Control",
    Eq. 2) 的 E = ½Jθ̇² + mgl(cosθ − 1) 相差一個常數 mgl：該文獻取 alpha=0
    （正上方）時 E=0 為參考零點，本函式則直接回傳未平移的物理能量，兩者的
    「能量誤差」在代數上完全等價（見 swing_up_voltage 推導）。
    """
    return (0.5 * PENDULUM_INERTIA * alpha_dot ** 2
            + PENDULUM_MASS * GRAVITY * PENDULUM_COM_RADIUS * math.cos(alpha))


def swing_up_voltage(alpha: float, alpha_dot: float) -> float:
    """
    能量法起擺（Åström & Furuta, 1996, Eq. 8 的飽和控制律）：

        u = sat_{nk}( k·(E − E0)·sign(alpha_dot·cos(alpha)) )

    文獻中 alpha=0（正上方）能量取 E0=0 為參考零點；本模組的 pendulum_energy()
    改用未平移的物理能量（相差常數 E_ref = Mp·g·lp），故 (E − E0) = −(E_ref − E)，
    展開後得到本函式實際計算的形式：

        voltage = sat( −μ·(E_ref − E)·sign(alpha_dot·cos(alpha)) )

    即 SWINGUP_DIRECTION_SIGN 的理論正確值為 -1.0（對應文獻的能量收斂方向），
    此為 config.py 的預設值。若實體硬體的編碼器 / 馬達接線極性相反導致起擺
    方向錯誤，才需要改為 +1.0（純屬硬體接線問題，與此處的能量控制推導無關）。
    """
    E_ref = PENDULUM_MASS * GRAVITY * PENDULUM_COM_RADIUS
    energy_error = E_ref - pendulum_energy(alpha, alpha_dot)

    direction = 1.0 if (alpha_dot * math.cos(alpha)) >= 0 else -1.0
    voltage = SWINGUP_DIRECTION_SIGN * SWINGUP_GAIN * energy_error * direction

    return max(-SWINGUP_VOLTAGE_LIMIT, min(SWINGUP_VOLTAGE_LIMIT, voltage))


def balance_voltage(theta: float, alpha: float,
                    theta_dot: float, alpha_dot: float,
                    voltage_limit: float) -> float:
    """LQR 平衡控制（參考點：theta=alpha=0，theta_dot=alpha_dot=0）。"""
    error   = -np.array([theta, alpha, theta_dot, alpha_dot])
    voltage = float(-np.dot(BALANCE_K, error))
    return max(-voltage_limit, min(voltage_limit, voltage))


def is_near_top(alpha: float, angle_tol_deg: float) -> bool:
    return abs(math.degrees(alpha)) <= angle_tol_deg


def is_switch_eligible(alpha: float, alpha_dot: float,
                       angle_tol_deg: float, rate_tol_rads: float) -> bool:
    """PRD 3.2：角度與角速度須同時落在容許範圍內，才算「穩定平衡」。"""
    return (abs(math.degrees(alpha)) <= angle_tol_deg
            and abs(alpha_dot) <= rate_tol_rads)
