## balance_control_qube_gui.py
# GUI-based balance control for the Qube Servo pendulum attachment.
# Supports virtual or physical Qube Servo 2 / Qube Servo 3.

# IF USING HARDWARE,  LIFT THE PENDULUM MANUALLY FOR THE CONTROLLER TO KICK IN
# IF USING VIRTUAL,   USE THE LIFT PENDULUM BUTTON IN QUANSER INTERACTIVE LABS
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

import csv
import math
import time
import threading
import collections
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pal.products.qube import QubeServo2, QubeServo3
from pal.utilities.math import ddt_filter

from constants import (
    FREQUENCY, DATA_RATE, GUI_RATE,
    BALANCE_THRESHOLD, VOLTAGE_LIMIT,
    THETA_DOT_CUTOFF, ALPHA_DOT_CUTOFF,
    PLOT_WINDOW, K_GAINS,
)

_BUF_SIZE = int(PLOT_WINDOW * DATA_RATE)


# ---------------------------------------------------------------------------
# ControlState — owns all mutable state for one run of the control loop.
# Creating a fresh instance on every Start avoids cross-run contamination.
# ---------------------------------------------------------------------------
class ControlState:
    def __init__(self):
        self.stop    = threading.Event()
        self._lock   = threading.Lock()
        self._times    = collections.deque(maxlen=_BUF_SIZE)
        self._alphas   = collections.deque(maxlen=_BUF_SIZE)
        self._thetas   = collections.deque(maxlen=_BUF_SIZE)
        self._voltages = collections.deque(maxlen=_BUF_SIZE)
        self.status  = 'Starting…'

    def append(self, t: float, alpha: float, theta: float, voltage: float) -> None:
        with self._lock:
            self._times.append(t)
            self._alphas.append(alpha)
            self._thetas.append(theta)
            self._voltages.append(voltage)

    def snapshot(self) -> tuple[list, list, list, list]:
        """Atomically copy all buffers; always returns four equal-length lists."""
        with self._lock:
            return (list(self._times), list(self._alphas),
                    list(self._thetas), list(self._voltages))

    def has_data(self) -> bool:
        return bool(self._times)


# ---------------------------------------------------------------------------
# Control loop  (background thread)
# ---------------------------------------------------------------------------
def control_loop(state: ControlState, qube_version: int,
                 hardware: int, pendulum: int, sim_time: int) -> None:
    dt        = 1.0 / FREQUENCY
    count_max = FREQUENCY / DATA_RATE
    count     = 0

    K         = K_GAINS[qube_version]
    QubeClass = QubeServo2 if qube_version == 2 else QubeServo3

    state_theta_dot = np.zeros(2, dtype=np.float64)
    state_alpha_dot = np.zeros(2, dtype=np.float64)

    try:
        with QubeClass(hardware=hardware, pendulum=pendulum, frequency=FREQUENCY) as qube:
            state.status = 'Running'
            start_time   = time.time()

            while not state.stop.is_set():
                qube.read_outputs()

                timestamp = time.time() - start_time
                if timestamp >= sim_time:
                    break

                theta   = qube.motorPosition * -1
                alpha_f = qube.pendulumPosition
                alpha   = np.mod(alpha_f, 2 * np.pi) - np.pi
                alpha_deg = abs(math.degrees(alpha))

                theta_dot, state_theta_dot = ddt_filter(
                    theta, state_theta_dot, THETA_DOT_CUTOFF, dt)
                alpha_dot, state_alpha_dot = ddt_filter(
                    alpha, state_alpha_dot, ALPHA_DOT_CUTOFF, dt)

                error = -np.array([theta, alpha, theta_dot, alpha_dot])

                if alpha_deg > BALANCE_THRESHOLD:
                    voltage = 0.0
                else:
                    voltage = float(np.clip(-np.dot(K, error), -VOLTAGE_LIMIT, VOLTAGE_LIMIT))

                qube.write_voltage(voltage)

                count += 1
                if count >= count_max:
                    state.append(timestamp, alpha, theta, voltage)
                    count = 0

        state.status = 'Finished'

    except Exception as exc:
        state.status = f'Error: {exc}'
    finally:
        state.stop.set()


# ---------------------------------------------------------------------------
# GUI application
# ---------------------------------------------------------------------------
_PLOT_CFG = [
    # (key,       title,                        ylabel,       color,     ylim)
    ('alpha',   'Pendulum angle — alpha',   'rad',        '#1f77b4', (-np.pi, np.pi)),
    ('theta',   'Base angle — theta',       'rad',        '#2ca02c', (-np.pi, np.pi)),
    ('voltage', 'Motor voltage',            'V',          '#d62728', (-VOLTAGE_LIMIT * 1.1,
                                                                       VOLTAGE_LIMIT * 1.1)),
]

_STATUS_COLORS = {
    'Idle':     'gray',
    'Running':  '#008800',
    'Finished': '#0055cc',
}


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Qube Servo — Balance Control')
        self.protocol('WM_DELETE_WINDOW', self._on_close)
        self._thread: threading.Thread | None = None
        self._state:  ControlState      | None = None
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
        self._setting_widgets: list[tk.Widget] = []
        for v in (2, 3):
            rb = ttk.Radiobutton(frm, text=str(v), variable=self._qube_ver, value=v)
            rb.pack(side='left')
            self._setting_widgets.append(rb)

        # Hardware mode
        ttk.Label(left, text='Mode:').grid(row=row, column=0, sticky='w', pady=4)
        self._hardware = tk.IntVar(value=1)
        frm = ttk.Frame(left); frm.grid(row=row, column=1, sticky='w'); row += 1
        for text, val in (('Virtual', 0), ('Physical', 1)):
            rb = ttk.Radiobutton(frm, text=text, variable=self._hardware, value=val)
            rb.pack(side='left')
            self._setting_widgets.append(rb)

        # Attachment (virtual only)
        ttk.Label(left, text='Attachment\n(virtual only):').grid(
            row=row, column=0, sticky='w', pady=4)
        self._pendulum = tk.IntVar(value=1)
        frm = ttk.Frame(left); frm.grid(row=row, column=1, sticky='w'); row += 1
        for text, val in (('DC Motor', 0), ('Pendulum', 1)):
            rb = ttk.Radiobutton(frm, text=text, variable=self._pendulum, value=val)
            rb.pack(side='left')
            self._setting_widgets.append(rb)

        # Simulation duration
        ttk.Label(left, text='Duration (s):').grid(row=row, column=0, sticky='w', pady=4)
        self._sim_time = tk.IntVar(value=30)
        sb = ttk.Spinbox(left, from_=5, to=600, increment=5,
                         textvariable=self._sim_time, width=7)
        sb.grid(row=row, column=1, sticky='w'); row += 1
        self._setting_widgets.append(sb)

        ttk.Separator(left, orient='horizontal').grid(
            row=row, column=0, columnspan=2, sticky='ew', pady=10); row += 1

        # Start / Stop / Export
        btn_frame = ttk.Frame(left)
        btn_frame.grid(row=row, column=0, columnspan=2, sticky='ew'); row += 1
        self._start_btn  = ttk.Button(btn_frame, text='Start',  command=self._start)
        self._stop_btn   = ttk.Button(btn_frame, text='Stop',   command=self._stop,
                                      state='disabled')
        self._export_btn = ttk.Button(btn_frame, text='Export CSV', command=self._export_csv,
                                      state='disabled')
        self._start_btn.pack( side='left', expand=True, fill='x')
        self._stop_btn.pack(  side='left', expand=True, fill='x', padx=(4, 0))
        self._export_btn.pack(side='left', expand=True, fill='x', padx=(4, 0))

        ttk.Separator(left, orient='horizontal').grid(
            row=row, column=0, columnspan=2, sticky='ew', pady=10); row += 1

        # Live readouts
        ttk.Label(left, text='Live values', font=('', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky='w'); row += 1
        mono = ('Courier', 10)
        self._lbl_alpha   = ttk.Label(left, text='alpha:    — rad', font=mono)
        self._lbl_theta   = ttk.Label(left, text='theta:    — rad', font=mono)
        self._lbl_voltage = ttk.Label(left, text='voltage:  — V',   font=mono)
        self._lbl_balance = ttk.Label(left, text='balance:  —',     font=mono)
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
        self._ax:    dict[str, object] = {}
        self._lines: dict[str, object] = {}

        for i, (key, title, ylabel, color, ylim) in enumerate(_PLOT_CFG, start=1):
            ax = fig.add_subplot(3, 1, i)
            ax.set_title(title, fontsize=9, loc='left')
            ax.set_xlabel('Time (s)', fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_xlim(0, PLOT_WINDOW)
            ax.set_ylim(*ylim)
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.axhline(0, color='k', linewidth=0.5, linestyle=':')
            self._lines[key], = ax.plot([], [], color=color, linewidth=1)
            self._ax[key] = ax

        # Balance-zone shading on alpha plot
        thresh_rad = math.radians(BALANCE_THRESHOLD)
        self._ax['alpha'].axhspan(
            -thresh_rad, thresh_rad,
            color='#1f77b4', alpha=0.08,
            label=f'±{BALANCE_THRESHOLD}° balance zone')
        self._ax['alpha'].legend(fontsize=7, loc='upper right')

        # Saturation lines on voltage plot
        for v in (VOLTAGE_LIMIT, -VOLTAGE_LIMIT):
            self._ax['voltage'].axhline(
                v, color='#d62728', linewidth=0.8, linestyle='--', alpha=0.6)

        canvas = FigureCanvasTkAgg(fig, master=right)
        canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')
        self._canvas = canvas

    # ---------------------------------------------------------------- actions
    def _set_settings_state(self, state: str) -> None:
        for w in self._setting_widgets:
            w.config(state=state)

    def _start(self):
        self._state = ControlState()
        self._set_settings_state('disabled')
        self._start_btn.config( state='disabled')
        self._stop_btn.config(  state='normal')
        self._export_btn.config(state='disabled')

        for key in self._lines:
            self._lines[key].set_data([], [])
        for ax in self._ax.values():
            ax.set_xlim(0, PLOT_WINDOW)

        self._thread = threading.Thread(
            target=control_loop,
            args=(self._state, self._qube_ver.get(), self._hardware.get(),
                  self._pendulum.get(), self._sim_time.get()),
            daemon=True,
        )
        self._thread.start()

    def _stop(self):
        if self._state:
            self._state.stop.set()

    def _on_close(self):
        if self._state:
            self._state.stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        self.destroy()

    def _export_csv(self):
        if not self._state or not self._state.has_data():
            messagebox.showinfo('Export', 'No data to export.')
            return
        path = filedialog.asksaveasfilename(
            defaultextension='.csv',
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')],
            title='Save data as CSV',
        )
        if not path:
            return
        t, alphas, thetas, voltages = self._state.snapshot()
        try:
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['time_s', 'alpha_rad', 'theta_rad', 'voltage_V'])
                writer.writerows(zip(t, alphas, thetas, voltages))
            messagebox.showinfo('Export', f'Saved {len(t)} rows to:\n{path}')
        except OSError as exc:
            messagebox.showerror('Export failed', str(exc))

    # -------------------------------------------------------- periodic refresh
    def _schedule_update(self):
        self._update()
        self.after(int(1000 / GUI_RATE), self._schedule_update)

    def _update(self):
        # Re-enable controls when thread finishes
        if self._thread is not None and not self._thread.is_alive():
            self._thread = None
            self._set_settings_state('normal')
            self._start_btn.config( state='normal')
            self._stop_btn.config(  state='disabled')
            if self._state and self._state.has_data():
                self._export_btn.config(state='normal')

        # Status label
        status = self._state.status if self._state else 'Idle'
        color  = _STATUS_COLORS.get(status.split(':')[0], '#cc0000')
        self._lbl_status.config(text=f'Status: {status}', foreground=color)

        if not self._state or not self._state.has_data():
            return

        # Atomic snapshot — guarantees equal-length lists for plotting
        t, alphas, thetas, voltages = self._state.snapshot()

        # Live value labels
        a, th, v = alphas[-1], thetas[-1], voltages[-1]
        self._lbl_alpha.config(  text=f'alpha:    {a:+.4f} rad  ({math.degrees(a):+.2f}°)')
        self._lbl_theta.config(  text=f'theta:    {th:+.4f} rad  ({math.degrees(th):+.2f}°)')
        self._lbl_voltage.config(text=f'voltage:  {v:+.3f} V')
        balancing = abs(math.degrees(a)) <= BALANCE_THRESHOLD
        self._lbl_balance.config(
            text=f'balance:  {"ACTIVE" if balancing else "inactive"}',
            foreground='#008800' if balancing else 'gray')

        if len(t) < 2:
            return

        # Redraw plots
        t_max = max(t[-1], PLOT_WINDOW)
        t_min = max(0.0, t[-1] - PLOT_WINDOW)

        buf_map = {'alpha': alphas, 'theta': thetas, 'voltage': voltages}
        for key, cfg in zip(buf_map, _PLOT_CFG):
            _, _, _, _, ylim = cfg
            vals = buf_map[key]
            self._lines[key].set_data(t, vals)
            ax = self._ax[key]
            ax.set_xlim(t_min, t_max)
            if key == 'voltage':
                ax.set_ylim(*ylim)   # fixed ±VOLTAGE_LIMIT*1.1
            else:
                span   = max(abs(v) for v in vals) if vals else 0.1
                margin = span * 0.2 or 0.1
                ax.set_ylim(-span - margin, span + margin)

        self._canvas.draw_idle()


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    app = App()
    app.mainloop()
