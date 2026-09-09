"""
tests/test_control.py — 核心控制邏輯的單元測試。

執行：
    python -m pytest tests/
"""
import math
import threading
import pytest

from control.filters   import LowPassFilter
from control.impedance import ImpedanceDynamics
from control.safety    import SafetyChecker, VoltageStepLimiter, saturate
from control.step      import StepContext, run_step
from control.pendulum  import (
    PendulumMode, DerivativeEstimator, wrap_angle,
    pendulum_energy, swing_up_voltage, balance_voltage,
    is_near_top, is_switch_eligible,
)
from config import (
    PENDULUM_MASS, GRAVITY, PENDULUM_COM_RADIUS,
    SWINGUP_VOLTAGE_LIMIT, PENDULUM_VOLTAGE_LIMIT,
    PENDULUM_ENGAGE_ANGLE_DEG, PENDULUM_ENGAGE_RATE_RADS,
)


# ── LowPassFilter ─────────────────────────────────────────────────────────────

class TestLowPassFilter:
    def test_zero_input_stays_zero(self):
        f = LowPassFilter(fc=20.0, dt=0.002)
        for _ in range(100):
            assert f.update(0.0) == pytest.approx(0.0)

    def test_step_converges(self):
        f = LowPassFilter(fc=20.0, dt=0.002)
        for _ in range(500):
            f.update(1.0)
        assert f.update(1.0) == pytest.approx(1.0, abs=1e-3)

    def test_dt_zero_safe(self):
        # dt=0 should not raise; alpha clamps via max(dt, 1e-9)
        f = LowPassFilter(fc=20.0, dt=0.0)
        assert math.isfinite(f.update(1.0))


# ── ImpedanceDynamics ─────────────────────────────────────────────────────────

class TestImpedanceDynamics:
    def test_zero_force_stays_at_origin(self):
        dyn = ImpedanceDynamics()
        dt = 0.002
        for _ in range(200):
            x = dyn.update(0.0, K=1.0, B=0.5, M=0.05, dt=dt)
        assert x == pytest.approx(0.0, abs=1e-6)

    def test_static_spring_mode(self):
        # M < 1e-6 → x = F_ext / K
        dyn = ImpedanceDynamics()
        x = dyn.update(2.0, K=4.0, B=0.1, M=0.0, dt=0.002)
        assert x == pytest.approx(0.5, abs=1e-6)

    def test_reset_clears_state(self):
        dyn = ImpedanceDynamics()
        for _ in range(100):
            dyn.update(1.0, K=1.0, B=0.1, M=0.05, dt=0.002)
        dyn.reset()
        x = dyn.update(0.0, K=1.0, B=0.1, M=0.05, dt=0.002)
        assert x == pytest.approx(0.0, abs=1e-9)

    def test_constant_force_settles(self):
        # With sufficient damping, x → F/K
        dyn = ImpedanceDynamics()
        K, B, M, F = 2.0, 2.0, 0.05, 1.0
        dt = 0.002
        for _ in range(5000):
            x = dyn.update(F, K=K, B=B, M=M, dt=dt)
        assert x == pytest.approx(F / K, abs=0.01)


# ── SafetyChecker ─────────────────────────────────────────────────────────────

class TestSafetyChecker:
    def _make_checker(self):
        logs = []
        emergency = threading.Event()
        checker = SafetyChecker(log_cb=logs.append, emergency_event=emergency)
        return checker, logs, emergency

    def test_normal_values_pass(self):
        checker, _, _ = self._make_checker()
        # skip warmup
        for _ in range(60):
            checker.check(0.0, 0.0, 0.0, 0.0)
        safe, reason = checker.check(0.1, 0.5, 0.5, 0.1)
        assert safe is True

    def test_angle_limit_triggers(self):
        checker, _, emergency = self._make_checker()
        safe, _ = checker.check(100.0, 0.0, 0.0, 0.0)
        assert safe is False
        assert emergency.is_set()

    def test_speed_limit_triggers(self):
        checker, _, emergency = self._make_checker()
        safe, _ = checker.check(0.0, 60.0, 0.0, 0.0)
        assert safe is False
        assert emergency.is_set()

    def test_current_skipped_during_warmup(self):
        checker, _, emergency = self._make_checker()
        # During warmup, large current should NOT trigger
        safe, _ = checker.check(0.0, 0.0, 100.0, 0.0)
        assert safe is True
        assert not emergency.is_set()

    def test_current_triggers_after_warmup(self):
        checker, _, emergency = self._make_checker()
        for _ in range(60):
            checker.check(0.0, 0.0, 0.0, 0.0)
        safe, _ = checker.check(0.0, 0.0, 100.0, 0.0)
        assert safe is False
        assert emergency.is_set()

    def test_force_zeroed_not_stopped(self):
        checker, _, emergency = self._make_checker()
        for _ in range(60):
            checker.check(0.0, 0.0, 0.0, 0.0)
        safe, reason = checker.check(0.0, 0.0, 0.0, 999.0)
        assert safe is True
        assert reason == "force_zero"
        assert not emergency.is_set()

    def test_reset_clears_cycle_count(self):
        checker, _, _ = self._make_checker()
        for _ in range(60):
            checker.check(0.0, 0.0, 0.0, 0.0)
        checker.reset()
        # After reset, current check is suppressed again (warmup restart)
        safe, _ = checker.check(0.0, 0.0, 100.0, 0.0)
        assert safe is True


# ── run_step ──────────────────────────────────────────────────────────────────

class TestRunStep:
    def _make_ctx(self, dt=0.002):
        dyn = ImpedanceDynamics()
        return StepContext(
            LowPassFilter(40, dt),
            LowPassFilter(15, dt),
            LowPassFilter(15, dt),
            dyn,
        )

    def _make_safety(self):
        emergency = threading.Event()
        checker = SafetyChecker(log_cb=lambda _: None,
                                emergency_event=emergency)
        # advance past warmup
        for _ in range(60):
            checker.check(0.0, 0.0, 0.0, 0.0)
        return checker

    def test_returns_four_tuple(self):
        ctx    = self._make_ctx()
        safety = self._make_safety()
        result = run_step(0.0, 0.0, ctx, (1.0, 0.1, 0.05, 20.0, 0.5, 0.0),
                          safety, 0.002, 1, 0.0)
        assert len(result) == 4

    def test_safe_step_produces_row_with_correct_keys(self):
        from config import ROW_FIELDS
        ctx    = self._make_ctx()
        safety = self._make_safety()
        safe, reason, voltage, row = run_step(
            0.0, 0.0, ctx, (1.0, 0.1, 0.05, 20.0, 0.5, 0.0),
            safety, 0.002, 1, 0.0)
        assert safe is True
        assert row is not None
        assert set(row.keys()) == set(ROW_FIELDS)

    def test_voltage_clamped(self):
        from config import VOLTAGE_LIMIT
        ctx    = self._make_ctx()
        safety = self._make_safety()
        # Very high Kp with large error → voltage should be clipped
        safe, _, voltage, _ = run_step(
            5.0, 0.0, ctx, (1.0, 0.1, 0.05, 1000.0, 0.0, 0.0),
            safety, 0.002, 1, 0.0)
        if safe:
            assert abs(voltage) <= VOLTAGE_LIMIT + 1e-9

    def test_unsafe_angle_returns_false(self):
        ctx    = self._make_ctx()
        safety = self._make_safety()
        safe, reason, voltage, row = run_step(
            100.0, 0.0, ctx, (1.0, 0.1, 0.05, 20.0, 0.5, 0.0),
            safety, 0.002, 1, 0.0)
        assert safe is False
        assert voltage == 0.0
        assert row is None


# ── VoltageStepLimiter / saturate ────────────────────────────────────────────

class TestVoltageStepLimiter:
    def test_saturate_clips(self):
        assert saturate(20.0, 10.0) == 10.0
        assert saturate(-20.0, 10.0) == -10.0
        assert saturate(3.0, 10.0) == 3.0

    def test_large_jump_is_clamped(self):
        lim = VoltageStepLimiter(max_step=1.0)
        lim.reset(0.0)
        out = lim.clamp(10.0)
        assert out == pytest.approx(1.0)

    def test_converges_over_multiple_cycles(self):
        lim = VoltageStepLimiter(max_step=1.0)
        lim.reset(0.0)
        out = 0.0
        for _ in range(10):
            out = lim.clamp(10.0)
        assert out == pytest.approx(10.0)

    def test_reset_sets_baseline(self):
        lim = VoltageStepLimiter(max_step=0.5)
        lim.reset(5.0)
        # Small step from the new baseline should pass through unclamped.
        assert lim.clamp(5.3) == pytest.approx(5.3)


# ── control/pendulum.py — wrap_angle ─────────────────────────────────────────

class TestWrapAngle:
    def test_zero_stays_zero(self):
        assert wrap_angle(0.0) == pytest.approx(0.0)

    def test_full_turn_wraps_to_zero(self):
        assert wrap_angle(2 * math.pi) == pytest.approx(0.0, abs=1e-9)

    def test_negative_full_turn_wraps_to_zero(self):
        assert wrap_angle(-2 * math.pi) == pytest.approx(0.0, abs=1e-9)

    def test_pi_and_a_half_turns_wrap_correctly(self):
        # 1.5 turns from 0 should land near -pi/2 (equivalent to -pi/2 mod 2pi)
        result = wrap_angle(3 * math.pi)
        assert abs(result) == pytest.approx(math.pi, abs=1e-9)


# ── control/pendulum.py — swing-up / balance / eligibility ─────────────────

class TestPendulumEnergy:
    def test_energy_zero_at_top_with_zero_velocity(self):
        # Matches Astrom & Furuta (1996) Eq. 2: E=0 at the top equilibrium.
        e = pendulum_energy(0.0, 0.0)
        assert e == pytest.approx(0.0)

    def test_energy_at_bottom_is_minus_two_mgl(self):
        # Matches Astrom & Furuta (1996): E=-2mgl in the downward position.
        e_bottom = pendulum_energy(math.pi, 0.0)
        assert e_bottom == pytest.approx(-2 * PENDULUM_MASS * GRAVITY * PENDULUM_COM_RADIUS)

    def test_energy_lower_at_bottom(self):
        e_top    = pendulum_energy(0.0, 0.0)
        e_bottom = pendulum_energy(math.pi, 0.0)
        assert e_bottom < e_top


class TestSwingUpVoltage:
    def test_voltage_bounded_by_swingup_limit(self):
        for alpha in (0.0, math.pi / 2, math.pi, -math.pi / 2):
            for alpha_dot in (-10.0, 0.0, 10.0):
                v = swing_up_voltage(alpha, alpha_dot)
                assert abs(v) <= SWINGUP_VOLTAGE_LIMIT + 1e-9

    def test_zero_at_exact_equilibrium(self):
        # At the top with zero velocity, energy error is zero → zero voltage.
        assert swing_up_voltage(0.0, 0.0) == pytest.approx(0.0)


class TestBalanceVoltage:
    def test_zero_state_gives_zero_voltage(self):
        v = balance_voltage(0.0, 0.0, 0.0, 0.0, PENDULUM_VOLTAGE_LIMIT)
        assert v == pytest.approx(0.0)

    def test_voltage_clamped_to_limit(self):
        v = balance_voltage(5.0, 5.0, 5.0, 5.0, PENDULUM_VOLTAGE_LIMIT)
        assert abs(v) <= PENDULUM_VOLTAGE_LIMIT + 1e-9

    def test_nonzero_alpha_produces_restoring_voltage(self):
        v = balance_voltage(0.0, 0.1, 0.0, 0.0, PENDULUM_VOLTAGE_LIMIT)
        assert v != 0.0


class TestSwitchEligibility:
    def test_near_top_within_tolerance(self):
        assert is_near_top(math.radians(3.0), angle_tol_deg=5.0) is True
        assert is_near_top(math.radians(8.0), angle_tol_deg=5.0) is False

    def test_switch_eligible_requires_both_angle_and_rate(self):
        assert is_switch_eligible(math.radians(2.0), 0.1,
                                  angle_tol_deg=5.0, rate_tol_rads=0.5) is True
        # Angle ok, rate too high
        assert is_switch_eligible(math.radians(2.0), 5.0,
                                  angle_tol_deg=5.0, rate_tol_rads=0.5) is False
        # Rate ok, angle too large
        assert is_switch_eligible(math.radians(10.0), 0.1,
                                  angle_tol_deg=5.0, rate_tol_rads=0.5) is False

    def test_fast_pass_through_top_is_not_engage_eligible(self):
        # Regression: real hardware log showed the pendulum flying past the
        # top at ~42 rad/s while within the +-10deg engage angle window; an
        # angle-only check wrongly treated that as "caught" and handed off
        # to the LQR mid-swing, which then fought the fast pendulum instead
        # of catching it -- preventing swing-up from ever succeeding.
        alpha = math.radians(-5.45)
        alpha_dot = 42.29
        assert is_near_top(alpha, PENDULUM_ENGAGE_ANGLE_DEG) is True
        assert is_switch_eligible(alpha, alpha_dot,
                                  PENDULUM_ENGAGE_ANGLE_DEG,
                                  PENDULUM_ENGAGE_RATE_RADS) is False

    def test_slow_near_top_is_engage_eligible(self):
        alpha = math.radians(3.0)
        alpha_dot = 0.5
        assert is_switch_eligible(alpha, alpha_dot,
                                  PENDULUM_ENGAGE_ANGLE_DEG,
                                  PENDULUM_ENGAGE_RATE_RADS) is True


# ── control/pendulum.py — DerivativeEstimator ───────────────────────────────

class TestDerivativeEstimator:
    def test_constant_signal_converges_to_zero_derivative(self):
        est = DerivativeEstimator(dt=0.002)
        for _ in range(500):
            theta_dot, alpha_dot = est.update(1.0, 0.5)
        assert theta_dot == pytest.approx(0.0, abs=1e-3)
        assert alpha_dot == pytest.approx(0.0, abs=1e-3)

    def test_reset_clears_history(self):
        est = DerivativeEstimator(dt=0.002)
        for _ in range(50):
            est.update(1.0, 1.0)
        est.reset()
        # First update after reset should not see a spike from the old history.
        theta_dot, alpha_dot = est.update(5.0, 5.0)
        assert theta_dot == pytest.approx(0.0, abs=1e-6)
        assert alpha_dot == pytest.approx(0.0, abs=1e-6)


class TestPendulumMode:
    def test_three_states_defined(self):
        assert {m.value for m in PendulumMode} == {"swingup", "balance", "impedance"}
