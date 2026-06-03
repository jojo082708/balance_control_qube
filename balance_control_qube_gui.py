## balance_control_qube_gui.py
# GUI-based balance control for the Qube Servo pendulum attachment.
# Supports virtual or physical Qube Servo 2 / Qube Servo 3.

# IF USING HARDWARE,  LIFT THE PENDULUM MANUALLY FOR THE CONTROLLER TO KICK IN
# IF USING VIRTUAL,   USE THE LIFT PENDULUM BUTTON IN QUANSER INTERACTIVE LABS
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

import math
import time
import threading
import collections
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pal.products.qube import QubeServo2, QubeServo3
from pal.utilities.math import ddt_filter

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
FREQUENCY         = 500     # control loop Hz
SCOPE_RATE        = 50      # GUI refresh / data-buffer Hz
BALANCE_THRESHOLD = 10.0    # degrees — deadzone before LQR activates
VOLTAGE_LIMIT     = 15.0    # volts  — hardware saturation limit
THETA_DOT_CUTOFF  = 50      # rad/s  — derivative filter cutoff for theta
ALPHA_DOT_CUTOFF  = 100     # rad/s  — derivative filter cutoff for alpha
PLOT_WINDOW       = 10      # seconds of history shown in plots

K_GAINS = {
    2: np.array([-1.0000, 34.7500, -1.4950,  3.1110]),
    3: np.array([-1.2247, 24.9044, -0.6877,  3.1321]),
}

# ---------------------------------------------------------------------------
# Thread-shared state
# (written only by control thread, read by GUI thread — CPython GIL makes
#  scalar/list element writes effectively atomic for these small types)
# ---------------------------------------------------------------------------
_stop_event  = threading.Event()
_buf_size    = int(PLOT_WINDOW * SCOPE_RATE)
_buf_time    = collections.deque(maxlen=_buf_size)
_buf_alpha   = collections.deque(maxlen=_buf_size)
_buf_theta   = collections.deque(maxlen=_buf_size)
_buf_voltage = collections.deque(maxlen=_buf_size)
_status      = ['Idle']   # single-element list acts as a mutable cell


# ---------------------------------------------------------------------------
# Control loop  (background thread)
# ---------------------------------------------------------------------------
def control_loop(qube_version: int, hardware: int, pendulum: int, sim_time: int):
    dt        = 1.0 / FREQUENCY
    count_max = FREQUENCY / SCOPE_RATE
    count     = 0

    K         = K_GAINS[qube_version]
    QubeClass = QubeServo2 if qube_version == 2 else QubeServo3

    state_theta_dot = np.zeros(2, dtype=np.float64)
    state_alpha_dot = np.zeros(2, dtype=np.float64)

    try:
        with QubeClass(hardware=hardware, pendulum=pendulum, frequency=FREQUENCY) as qube:
            _status[0] = 'Running'
            start_time = time.time()

            while not _stop_event.is_set():
                qube.read_outputs()

                timestamp = time.time() - start_time
                if timestamp >= sim_time:
                    break

                # State estimation
                theta   = qube.motorPosition * -1
                alpha_f = qube.pendulumPosition
                alpha   = np.mod(alpha_f, 2 * np.pi) - np.pi
                alpha_deg = abs(math.degrees(alpha))

                theta_dot, state_theta_dot = ddt_filter(
                    theta, state_theta_dot, THETA_DOT_CUTOFF, dt)
                alpha_dot, state_alpha_dot = ddt_filter(
                    alpha, state_alpha_dot, ALPHA_DOT_CUTOFF, dt)

                # LQR (error = 0 - state since reference is zero)
                error = -np.array([theta, alpha, theta_dot, alpha_dot])

                if alpha_deg > BALANCE_THRESHOLD:
                    voltage = 0.0
                else:
                    voltage = float(np.clip(-np.dot(K, error), -VOLTAGE_LIMIT, VOLTAGE_LIMIT))

                qube.write_voltage(voltage)

                count += 1
                if count >= count_max:
                    _buf_time.append(timestamp)
                    _buf_alpha.append(alpha)
                    _buf_theta.append(theta)
                    _buf_voltage.append(voltage)
                    count = 0

        _status[0] = 'Finished'

    except Exception as exc:
        _status[0] = f'Error: {exc}'
    finally:
        _stop_event.set()


# ---------------------------------------------------------------------------
# GUI application
# ---------------------------------------------------------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Qube Servo — Balance Control')
        self.protocol('WM_DELETE_WINDOW', self._on_close)
        self._thread: threading.Thread | None = None
        self._build_ui()
        self._schedule_update()

    # ------------------------------------------------------------------ build
    def _build_ui(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # ---- Left panel ------------------------------------------------
        left = ttk.Frame(self, padding=12)
        left.grid(row=0, column=0, sticky='ns')
        left.columnconfigure(1, weight=1)

        row = 0

        ttk.Label(left, text='Settings', font=('', 11, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky='w', pady=(0, 8)); row += 1

        # Qube version
        ttk.Label(left, text='Qube version:').grid(row=row, column=0, sticky='w')
        self._qube_ver = tk.IntVar(value=3)
        frm = ttk.Frame(left); frm.grid(row=row, column=1, sticky='w'); row += 1
        for v in (2, 3):
            ttk.Radiobutton(frm, text=str(v), variable=self._qube_ver, value=v).pack(side='left')

        # Hardware mode
        ttk.Label(left, text='Mode:').grid(row=row, column=0, sticky='w', pady=4)
        self._hardware = tk.IntVar(value=1)
        frm = ttk.Frame(left); frm.grid(row=row, column=1, sticky='w'); row += 1
        ttk.Radiobutton(frm, text='Virtual',  variable=self._hardware, value=0).pack(side='left')
        ttk.Radiobutton(frm, text='Physical', variable=self._hardware, value=1).pack(side='left')

        # Attachment (virtual only)
        ttk.Label(left, text='Attachment\n(virtual only):').grid(
            row=row, column=0, sticky='w', pady=4)
        self._pendulum = tk.IntVar(value=1)
        frm = ttk.Frame(left); frm.grid(row=row, column=1, sticky='w'); row += 1
        ttk.Radiobutton(frm, text='DC Motor',  variable=self._pendulum, value=0).pack(side='left')
        ttk.Radiobutton(frm, text='Pendulum',  variable=self._pendulum, value=1).pack(side='left')

        # Simulation duration
        ttk.Label(left, text='Duration (s):').grid(row=row, column=0, sticky='w', pady=4)
        self._sim_time = tk.IntVar(value=30)
        ttk.Spinbox(left, from_=5, to=600, increment=5,
                    textvariable=self._sim_time, width=7).grid(
            row=row, column=1, sticky='w'); row += 1

        ttk.Separator(left, orient='horizontal').grid(
            row=row, column=0, columnspan=2, sticky='ew', pady=10); row += 1

        # Start / Stop buttons
        btn_frame = ttk.Frame(left)
        btn_frame.grid(row=row, column=0, columnspan=2, sticky='ew'); row += 1
        self._start_btn = ttk.Button(btn_frame, text='Start', command=self._start)
        self._start_btn.pack(side='left', expand=True, fill='x')
        self._stop_btn  = ttk.Button(btn_frame, text='Stop',
                                     command=self._stop, state='disabled')
        self._stop_btn.pack(side='left', expand=True, fill='x', padx=(6, 0))

        ttk.Separator(left, orient='horizontal').grid(
            row=row, column=0, columnspan=2, sticky='ew', pady=10); row += 1

        # Live readouts
        ttk.Label(left, text='Live values', font=('', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky='w'); row += 1
        self._lbl_alpha   = ttk.Label(left, text='alpha:    — rad', font=('Courier', 10))
        self._lbl_theta   = ttk.Label(left, text='theta:    — rad', font=('Courier', 10))
        self._lbl_voltage = ttk.Label(left, text='voltage:  — V',   font=('Courier', 10))
        self._lbl_balance = ttk.Label(left, text='balance:  —',     font=('Courier', 10))
        for w in (self._lbl_alpha, self._lbl_theta, self._lbl_voltage, self._lbl_balance):
            w.grid(row=row, column=0, columnspan=2, sticky='w', pady=1); row += 1

        ttk.Separator(left, orient='horizontal').grid(
            row=row, column=0, columnspan=2, sticky='ew', pady=10); row += 1

        self._lbl_status = ttk.Label(left, text='Status: Idle', foreground='gray')
        self._lbl_status.grid(row=row, column=0, columnspan=2, sticky='w'); row += 1

        # ---- Right panel: plots ----------------------------------------
        right = ttk.Frame(self, padding=(0, 10, 10, 10))
        right.grid(row=0, column=1, sticky='nsew')
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        fig = Figure(figsize=(9, 6), tight_layout=True)
        self._ax = {
            'alpha':   fig.add_subplot(3, 1, 1),
            'theta':   fig.add_subplot(3, 1, 2),
            'voltage': fig.add_subplot(3, 1, 3),
        }

        plot_cfg = [
            ('alpha',   'Pendulum angle — alpha (rad)', '#1f77b4', (-np.pi, np.pi)),
            ('theta',   'Base angle — theta (rad)',      '#2ca02c', (-np.pi, np.pi)),
            ('voltage', 'Motor voltage (V)',              '#d62728', (-VOLTAGE_LIMIT, VOLTAGE_LIMIT)),
        ]
        self._lines = {}
        for key, title, color, ylim in plot_cfg:
            ax = self._ax[key]
            ax.set_title(title, fontsize=9, loc='left')
            ax.set_xlabel('Time (s)', fontsize=8)
            ax.set_ylabel(ax.get_title().split('(')[1].rstrip(')') if '(' in title else '',
                          fontsize=8)
            ax.set_xlim(0, PLOT_WINDOW)
            ax.set_ylim(*ylim)
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.axhline(0, color='k', linewidth=0.5, linestyle=':')
            self._lines[key], = ax.plot([], [], color=color, linewidth=1)

        # Threshold band on alpha plot
        self._ax['alpha'].axhspan(
            -math.radians(BALANCE_THRESHOLD), math.radians(BALANCE_THRESHOLD),
            color='#1f77b4', alpha=0.08, label=f'±{BALANCE_THRESHOLD}° balance zone')
        self._ax['alpha'].legend(fontsize=7, loc='upper right')

        # Saturation lines on voltage plot
        for v in (VOLTAGE_LIMIT, -VOLTAGE_LIMIT):
            self._ax['voltage'].axhline(v, color='#d62728', linewidth=0.8,
                                        linestyle='--', alpha=0.6)

        canvas = FigureCanvasTkAgg(fig, master=right)
        canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')
        self._canvas = canvas

    # ---------------------------------------------------------------- actions
    def _start(self):
        _stop_event.clear()
        for buf in (_buf_time, _buf_alpha, _buf_theta, _buf_voltage):
            buf.clear()
        for key in self._lines:
            self._lines[key].set_data([], [])
        for ax in self._ax.values():
            ax.set_xlim(0, PLOT_WINDOW)
        _status[0] = 'Starting…'

        self._thread = threading.Thread(
            target=control_loop,
            args=(self._qube_ver.get(), self._hardware.get(),
                  self._pendulum.get(), self._sim_time.get()),
            daemon=True,
        )
        self._thread.start()
        self._start_btn.config(state='disabled')
        self._stop_btn.config(state='normal')

    def _stop(self):
        _stop_event.set()

    def _on_close(self):
        _stop_event.set()
        self.destroy()

    # -------------------------------------------------------- periodic refresh
    def _schedule_update(self):
        self._update()
        self.after(int(1000 / SCOPE_RATE), self._schedule_update)

    def _update(self):
        # Re-enable Start when thread finishes
        if self._thread is not None and not self._thread.is_alive():
            self._thread = None
            self._start_btn.config(state='normal')
            self._stop_btn.config(state='disabled')

        # Status label
        status = _status[0]
        color = {'Idle': 'gray', 'Running': '#008800',
                 'Finished': '#0055cc'}.get(
            status.split(':')[0], '#cc0000')
        self._lbl_status.config(text=f'Status: {status}', foreground=color)

        if not _buf_time:
            return

        # Live value labels
        a, th, v = _buf_alpha[-1], _buf_theta[-1], _buf_voltage[-1]
        self._lbl_alpha.config(  text=f'alpha:    {a:+.4f} rad  ({math.degrees(a):+.2f}°)')
        self._lbl_theta.config(  text=f'theta:    {th:+.4f} rad  ({math.degrees(th):+.2f}°)')
        self._lbl_voltage.config(text=f'voltage:  {v:+.3f} V')
        balancing = abs(math.degrees(a)) <= BALANCE_THRESHOLD
        self._lbl_balance.config(
            text=f'balance:  {"ACTIVE" if balancing else "inactive"}',
            foreground='#008800' if balancing else 'gray')

        # Redraw plots
        if len(_buf_time) < 2:
            return
        t = list(_buf_time)
        t_min = max(0.0, t[-1] - PLOT_WINDOW)
        t_max = max(t[-1], PLOT_WINDOW)

        data = {
            'alpha':   list(_buf_alpha),
            'theta':   list(_buf_theta),
            'voltage': list(_buf_voltage),
        }
        fixed_ylim = {
            'voltage': (-VOLTAGE_LIMIT * 1.1, VOLTAGE_LIMIT * 1.1),
        }
        for key, vals in data.items():
            self._lines[key].set_data(t, vals)
            ax = self._ax[key]
            ax.set_xlim(t_min, t_max)
            if key in fixed_ylim:
                ax.set_ylim(*fixed_ylim[key])
            else:
                span = max(abs(v) for v in vals) if vals else 0.1
                margin = span * 0.2 or 0.1
                ax.set_ylim(-span - margin, span + margin)

        self._canvas.draw_idle()


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    app = App()
    app.mainloop()
