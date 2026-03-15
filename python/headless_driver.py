"""
headless_driver.py
Calls the C++ engine via ctypes, collects advertisement data,
and generates a 6-panel summary plot.

Build first:
  g++ -O3 -shared -fPIC -std=c++11 \
      -o cpp/libheuristic.so \
      cpp/engine.cpp cpp/simulator.cpp cpp/learner.cpp \
      -Icpp
"""

import ctypes
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless – no display needed
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ─── Load shared library ──────────────────────────────────────────────────────
def _load_library():
    candidates = [
        "./cpp/libheuristic.so",    # Linux / Colab
        "./cpp/libheuristic.dylib", # macOS
        "./cpp/heuristic.dll",      # Windows
    ]
    for path in candidates:
        if os.path.exists(path):
            return ctypes.CDLL(path)
    sys.exit(
        "ERROR: shared library not found.\n"
        "Build with:\n"
        "  g++ -O3 -shared -fPIC -std=c++11 \\\n"
        "      -o cpp/libheuristic.so \\\n"
        "      cpp/engine.cpp cpp/simulator.cpp cpp/learner.cpp -Icpp"
    )

lib = _load_library()

# ─── ctypes signatures ────────────────────────────────────────────────────────
lib.create_engine.argtypes  = [ctypes.c_char_p]
lib.create_engine.restype   = ctypes.c_void_p

lib.destroy_engine.argtypes = [ctypes.c_void_p]
lib.destroy_engine.restype  = None

# timestamp, mac, uid, service, rssi, x, y, is_rogue, rogue_type, logical_id, anomaly
CALLBACK_TYPE = ctypes.CFUNCTYPE(
    None,
    ctypes.c_double,   # timestamp
    ctypes.c_char_p,   # mac
    ctypes.c_char_p,   # uid
    ctypes.c_char_p,   # service_id
    ctypes.c_double,   # rssi
    ctypes.c_double,   # x
    ctypes.c_double,   # y
    ctypes.c_int,      # is_rogue
    ctypes.c_char_p,   # rogue_type
    ctypes.c_int,      # logical_id
    ctypes.c_int,      # anomaly
)

lib.set_callback.argtypes   = [ctypes.c_void_p, CALLBACK_TYPE]
lib.set_callback.restype    = None

lib.run_step.argtypes       = [ctypes.c_void_p, ctypes.c_double]
lib.run_step.restype        = ctypes.c_int

lib.get_stats_json.argtypes = [ctypes.c_void_p]
lib.get_stats_json.restype  = ctypes.c_char_p

lib.update_thresholds.argtypes = [ctypes.c_void_p,
                                   ctypes.c_double,
                                   ctypes.c_double,
                                   ctypes.c_double]
lib.update_thresholds.restype  = None

# ─── Data collector ───────────────────────────────────────────────────────────
# FIX: use a mutable container (dict of lists) so the callback can append to it
# and the main script can later convert to numpy arrays without shadowing issues.
class DataStore:
    def __init__(self):
        self.timestamps   = []
        self.logical_ids  = []
        self.anomalies    = []
        self.rssi_vals    = []
        self.rogue_truth  = []
        self.th_history   = []   # list of (time, rssi_th, int_th, sim_th)

store = DataStore()

# ─── Callback ─────────────────────────────────────────────────────────────────
@CALLBACK_TYPE
def advert_callback(ts, mac, uid, service, rssi, x, y,
                    is_rogue, rogue_type, logical_id, anomaly):
    store.timestamps.append(ts)
    store.logical_ids.append(logical_id)
    store.anomalies.append(anomaly)
    store.rssi_vals.append(rssi)
    store.rogue_truth.append(is_rogue)

# ─── Simulation runner ────────────────────────────────────────────────────────
def run_simulation(config: dict, real_time_sec: float = 30.0):
    """
    Drives the C++ engine for `real_time_sec` seconds of wall-clock steps
    (each step advances simulation by dt seconds).
    Returns the final stats dict.
    """
    config_json = json.dumps(config).encode("utf-8")
    engine = lib.create_engine(config_json)
    lib.set_callback(engine, advert_callback)

    dt    = 0.1          # simulation timestep (seconds)
    steps = int(real_time_sec / dt)
    print(f"Running {steps} steps (dt={dt}s)…")

    for step in range(steps):
        ret = lib.run_step(engine, dt)
        if ret == 0:
            print(f"  Simulation finished early at step {step}.")
            break

        # Collect threshold history every 100 steps
        if step % 100 == 0:
            raw   = lib.get_stats_json(engine)
            stats = json.loads(raw.decode("utf-8"))
            store.th_history.append((
                stats["time"],
                stats["rssi_th"],
                stats["int_th"],
                stats["sim_th"],
            ))

    raw        = lib.get_stats_json(engine)
    final_stats = json.loads(raw.decode("utf-8"))
    lib.destroy_engine(engine)
    return final_stats

# ─── Plotting ─────────────────────────────────────────────────────────────────
def generate_plots(outfile: str = "simulation_results.png"):
    # FIX: convert from store – no global-list shadow issue
    ts      = np.array(store.timestamps,  dtype=float)
    lids    = np.array(store.logical_ids, dtype=int)
    anom    = np.array(store.anomalies,   dtype=int)
    rssi    = np.array(store.rssi_vals,   dtype=float)
    truth   = np.array(store.rogue_truth, dtype=int)

    th_times, th_rssi, th_int, th_sim = [], [], [], []
    for (t, r, i, s) in store.th_history:
        th_times.append(t); th_rssi.append(r)
        th_int.append(i);   th_sim.append(s)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("BLE Beacon Anomaly Detection — Simulation Results", fontsize=14)

    # 1. RSSI over time (colour by ground truth)
    ax = axes[0, 0]
    normal_mask = truth == 0
    rogue_mask  = truth == 1
    ax.plot(ts[normal_mask], rssi[normal_mask], "b.", markersize=1, label="Normal", alpha=0.5)
    ax.plot(ts[rogue_mask],  rssi[rogue_mask],  "r.", markersize=2, label="Rogue",  alpha=0.8)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("RSSI (dBm)")
    ax.set_title("RSSI over Time");  ax.legend(markerscale=4)

    # 2. Anomaly detections over time
    ax = axes[0, 1]
    ax.plot(ts, anom, "r.", markersize=1, alpha=0.4)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Anomaly flag (0/1)")
    ax.set_title("Detected Anomalies")

    # 3. Adaptive thresholds over time
    ax = axes[1, 0]
    if th_times:
        ax.plot(th_times, th_rssi, label="RSSI threshold")
        ax.plot(th_times, th_int,  label="Interval threshold")
        ax.plot(th_times, th_sim,  label="Similarity threshold")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Threshold value")
        ax.set_title("Adaptive Threshold Evolution"); ax.legend()
    else:
        ax.set_title("Adaptive Thresholds (no data)")

    # 4. Advertisements per logical ID
    ax = axes[1, 1]
    if len(lids) > 0:
        unique, counts = np.unique(lids, return_counts=True)
        ax.bar(unique, counts, color="steelblue")
        ax.set_xlabel("Logical device ID"); ax.set_ylabel("Advertisement count")
        ax.set_title("Advertisements per Logical Device")

    # 5. Confusion matrix (predicted anomaly vs ground-truth rogue)
    ax = axes[2, 0]
    if len(truth) > 0 and len(np.unique(truth)) > 1:
        cm   = confusion_matrix(truth, anom)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=["Normal", "Rogue"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title("Anomaly Detection — Confusion Matrix")
    else:
        ax.set_title("Confusion matrix (need both classes)")

    # 6. Rolling anomaly rate
    ax = axes[2, 1]
    window = min(1000, len(anom) // 5)
    if window > 0 and len(anom) > window:
        rate = np.convolve(anom.astype(float),
                           np.ones(window) / window, mode="valid")
        ax.plot(ts[window - 1:], rate, "g-", linewidth=1)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Anomaly rate")
        ax.set_title(f"Rolling Anomaly Rate (window={window})")
    else:
        ax.set_title("Rolling anomaly rate (insufficient data)")

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {outfile}")

# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    config = {
        "num_static":    10,
        "num_mobile":     5,
        "rogue_percent": 10.0,
        "duration_hours": 0.5,
        "width":        100.0,
        "height":       100.0,
        "rssi_th":       10.0,
        "int_th":         0.1,
        "sim_th":         0.8,
    }

    print("=== Adaptive BLE Beacon Heuristic Engine ===")
    final = run_simulation(config, real_time_sec=30.0)
    print("Final stats:", json.dumps(final, indent=2))
    print(f"Total advertisements collected: {len(store.timestamps)}")

    if len(store.timestamps) == 0:
        print("WARNING: no advertisements were collected – "
              "check that the library compiled correctly.")
    else:
        generate_plots("simulation_results.png")
