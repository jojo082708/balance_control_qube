## constants.py
# Shared configuration for balance_control_qube scripts.
import numpy as np

FREQUENCY         = 500    # Hz  — hardware control loop rate
DATA_RATE         = 50     # Hz  — sensor capture / Scope sample rate
GUI_RATE          = 25     # Hz  — GUI redraw rate (independent of data capture)
BALANCE_THRESHOLD = 10.0   # deg — deadzone before LQR activates
VOLTAGE_LIMIT     = 15.0   # V   — output saturation limit
THETA_DOT_CUTOFF  = 50     # rad/s — derivative filter cutoff for theta
ALPHA_DOT_CUTOFF  = 100    # rad/s — derivative filter cutoff for alpha
PLOT_WINDOW       = 10     # s   — scrolling plot history length
THETA_LIMIT       = 3.14159265358979  # rad — arm safety stop (≈ ±180°, adjust per hardware)

K_GAINS = {
    2: np.array([-1.0000, 34.7500, -1.4950,  3.1110]),
    3: np.array([-1.2247, 24.9044, -0.6877,  3.1321]),
}
