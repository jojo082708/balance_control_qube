## balance_control_qube.py
# Balance control of the Qube Servo's Pendulum attachment.
# Supports virtual or physical Qube Servo 2 / Qube Servo 3 in task-based (time-based IO) mode.

# IF USING HARDWARE,  LIFT THE PENDULUM MANUALLY FOR THE CONTROLLER TO KICK IN
# IF USING VIRTUAL,   USE THE LIFT PENDULUM BUTTON IN QUANSER INTERACTIVE LABS
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

import signal
import time
import math
import threading
import numpy as np
from pal.products.qube import QubeServo2, QubeServo3
from pal.utilities.math import ddt_filter
from pal.utilities.scope import Scope

from constants import (
    FREQUENCY, DATA_RATE, BALANCE_THRESHOLD, VOLTAGE_LIMIT,
    THETA_DOT_CUTOFF, ALPHA_DOT_CUTOFF, K_GAINS,
)

# ---------------------------------------------------------------------------
# Script-level config
# ---------------------------------------------------------------------------
SIMULATION_TIME = 30  # seconds

# ---------------------------------------------------------------------------
# Graceful shutdown via Ctrl-C
# ---------------------------------------------------------------------------
_stop_event = threading.Event()

def _sig_handler(*_args):
    _stop_event.set()

signal.signal(signal.SIGINT, _sig_handler)

# ---------------------------------------------------------------------------
# Scopes
# ---------------------------------------------------------------------------
scopePendulum = Scope(
    title='Pendulum angle - alpha (rad)',
    timeWindow=10,
    xLabel='Time (s)',
    yLabel='Position (rad)')
scopePendulum.attachSignal(name='alpha (rad)', width=1)

scopeBase = Scope(
    title='Base angle - theta (rad)',
    timeWindow=10,
    xLabel='Time (s)',
    yLabel='Position (rad)')
scopeBase.attachSignal(name='theta (rad)', width=1)

scopeVoltage = Scope(
    title='Motor Voltage',
    timeWindow=10,
    xLabel='Time (s)',
    yLabel='Voltage (V)')
scopeVoltage.attachSignal(name='Voltage (V)', width=1)


# ---------------------------------------------------------------------------
# Control loop
# ---------------------------------------------------------------------------
def control_loop():
    # CHANGE THESE FOR YOUR SETUP
    qube_version = 3   # 2 or 3
    hardware     = 1   # 0 = virtual, 1 = physical
    # Virtual only: 0 = DC motor attachment, 1 = pendulum attachment
    pendulum     = 1

    dt        = 1.0 / FREQUENCY
    count_max = FREQUENCY / DATA_RATE
    count     = 0

    K         = K_GAINS[qube_version]
    QubeClass = QubeServo2 if qube_version == 2 else QubeServo3

    state_theta_dot = np.zeros(2, dtype=np.float64)
    state_alpha_dot = np.zeros(2, dtype=np.float64)

    try:
        with QubeClass(hardware=hardware, pendulum=pendulum, frequency=FREQUENCY) as qube:
            start_time = time.time()

            while not _stop_event.is_set():
                qube.read_outputs()

                timestamp = time.time() - start_time
                if timestamp >= SIMULATION_TIME:
                    break

                # --- State estimation ---
                theta   = qube.motorPosition * -1
                alpha_f = qube.pendulumPosition
                alpha   = np.mod(alpha_f, 2 * np.pi) - np.pi
                alpha_deg = abs(math.degrees(alpha))

                theta_dot, state_theta_dot = ddt_filter(
                    theta, state_theta_dot, THETA_DOT_CUTOFF, dt)
                alpha_dot, state_alpha_dot = ddt_filter(
                    alpha, state_alpha_dot, ALPHA_DOT_CUTOFF, dt)

                # --- LQR (reference = 0 for all states) ---
                error = -np.array([theta, alpha, theta_dot, alpha_dot])

                if alpha_deg > BALANCE_THRESHOLD:
                    voltage = 0.0
                else:
                    voltage = float(np.clip(-np.dot(K, error), -VOLTAGE_LIMIT, VOLTAGE_LIMIT))

                qube.write_voltage(voltage)

                # --- Scope update (rate-limited to DATA_RATE) ---
                count += 1
                if count >= count_max:
                    scopePendulum.sample(timestamp, [alpha])
                    scopeBase.sample(timestamp, [theta])
                    scopeVoltage.sample(timestamp, [voltage])
                    count = 0

    except Exception as exc:
        print(f'\n[control_loop] Error: {exc}')
    finally:
        _stop_event.set()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
thread = threading.Thread(target=control_loop, daemon=True)
thread.start()

# Keep refreshing scopes until the control thread actually finishes.
# Looping on thread.is_alive() (not _stop_event) ensures a final refresh
# happens during the thread's finally-block cleanup.
while thread.is_alive():
    Scope.refreshAll()
    time.sleep(0.01)

input('Press Enter to exit.')
