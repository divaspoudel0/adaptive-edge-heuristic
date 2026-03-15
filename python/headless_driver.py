"""
headless_driver.py
100-hour BLE beacon simulation with large advertisement volume.
Advertisements are streamed directly to CSV to avoid RAM exhaustion.

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
import csv
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ─── Paths ────────────────────────────────────────────────────────────────────
CSV_FILE  = "advertisements.csv"
PLOT_FILE = "simulation_results.png"

# ─── Load shared library ──────────────────────────────────────────────────────
def _load_library():
    candidates = [
        "./cpp/libheuristic.so",
        "./cpp/libheuristic.dylib",
        "./cpp/heuristic.dll",
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

lib.set_callback.argtypes      = [ctypes.c_void_p, CALLBACK_TYPE]
lib.set_callback.restype       = None
lib.run_step.argtypes          = [ctypes.c_void_p, ctypes.c_double]
lib.run_step.restype           = ctypes.c_int
lib.get_stats_json.argtypes    = [ctypes.c_void_p]
lib.get_stats_json.restype     = ctypes.c_char_p
lib.update_thresholds.argtypes = [ctypes.c_void_p,
                                   ctypes.c_double,
                                   ctypes.c_double,
                                   ctypes.c_double]
lib.update_thresholds.restype  = None

# ─── Streaming CSV writer ─────────────────────────────────────────────────────
# Writes every advertisement to disk to avoid RAM exhaustion.
# Keeps a 1-in-SAMPLE_EVERY sparse sample in RAM for plotting only.

SAMPLE_EVERY = 500       # 1 in 500 ads kept in RAM for plots
FLUSH_EVERY  = 50_000    # flush CSV to disk every N rows

class StreamingStore:
    def __init__(self, csv_path: str):
        self.csv_path  = csv_path
        self._fh       = open(csv_path, "w", newline="", buffering=1)
        self._writer   = csv.writer(self._fh)
        self._writer.writerow([
            "timestamp", "mac", "uid", "service_id",
            "rssi", "x", "y", "is_rogue", "rogue_type",
            "logical_id", "anomaly"
        ])
        # Sparse in-RAM sample for plotting
        self.s_ts    = []
        self.s_lids  = []
        self.s_anom  = []
        self.s_rssi  = []
        self.s_truth = []

        # Threshold snapshots
        self.th_history = []   # [(sim_time, rssi_th, int_th, sim_th)]

        self.total_count   = 0
        self.rogue_count   = 0
        self.anomaly_count = 0
        self._buf_count    = 0

    def record(self, ts, mac, uid, service, rssi, x, y,
               is_rogue, rogue_type, logical_id, anomaly):
        self._writer.writerow([
            f"{ts:.3f}", mac, uid, service,
            f"{rssi:.2f}", f"{x:.2f}", f"{y:.2f}",
            is_rogue, rogue_type, logical_id, anomaly
        ])
        self.total_count   += 1
        self.rogue_count   += is_rogue
        self.anomaly_count += anomaly
        self._buf_count    += 1

        if self._buf_count >= FLUSH_EVERY:
            self._fh.flush()
            self._buf_count = 0

        # Keep sparse sample for plotting
        if self.total_count % SAMPLE_EVERY == 0:
            self.s_ts.append(ts)
            self.s_lids.append(logical_id)
            self.s_anom.append(anomaly)
            self.s_rssi.append(rssi)
            self.s_truth.append(is_rogue)

    def close(self):
        self._fh.flush()
        self._fh.close()

store = StreamingStore(CSV_FILE)

# ─── Callback ─────────────────────────────────────────────────────────────────
@CALLBACK_TYPE
def advert_callback(ts, mac, uid, service, rssi, x, y,
                    is_rogue, rogue_type, logical_id, anomaly):
    store.record(
        ts,
        mac.decode("utf-8"),
        uid.decode("utf-8"),
        service.decode("utf-8"),
        rssi, x, y,
        is_rogue,
        rogue_type.decode("utf-8"),
        logical_id,
        anomaly,
    )

# ─── Simulation runner ────────────────────────────────────────────────────────
def run_simulation(config: dict):
    sim_seconds  = config["duration_hours"] * 3600.0
    dt           = 0.5                          # 0.5s step keeps step count manageable
    total_steps  = int(sim_seconds / dt)
    report_every = max(1, int(3600.0 / dt))     # progress every 1 sim-hour

    config_json = json.dumps(config).encode("utf-8")
    engine      = lib.create_engine(config_json)
    lib.set_callback(engine, advert_callback)

    print(f"Simulation : {config['duration_hours']} hours")
    print(f"Steps      : {total_steps:,}  (dt={dt}s)")
    print(f"Devices    : {config['num_static']} static + "
          f"{config['num_mobile']} mobile")
    print(f"CSV output : {CSV_FILE}\n")

    wall_start = time.time()

    for step in range(total_steps):
        ret = lib.run_step(engine, dt)
        if ret == 0:
            print(f"\n  Simulation finished early at step {step:,}.")
            break

        # Snapshot thresholds every 200 steps
        if step % 200 == 0:
            raw   = lib.get_stats_json(engine)
            stats = json.loads(raw.decode("utf-8"))
            store.th_history.append((
                stats["time"],
                stats["rssi_th"],
                stats["int_th"],
                stats["sim_th"],
            ))

        # Progress report every sim-hour
        if step > 0 and step % report_every == 0:
            sim_hr   = (step * dt) / 3600.0
            elapsed  = time.time() - wall_start
            rate     = store.total_count / max(elapsed, 1)
            pct      = 100.0 * step / total_steps
            eta_sec  = (total_steps - step) * dt / max(rate * dt, 1e-9)
            print(f"  [{pct:5.1f}%]  sim={sim_hr:6.1f}h  "
                  f"ads={store.total_count:>13,}  "
                  f"rogues={store.rogue_count:>9,}  "
                  f"anomalies={store.anomaly_count:>9,}  "
                  f"wall={elapsed:.0f}s  "
                  f"rate={rate:,.0f} ads/s  "
                  f"ETA={eta_sec/60:.1f}min")

    raw         = lib.get_stats_json(engine)
    final_stats = json.loads(raw.decode("utf-8"))
    lib.destroy_engine(engine)
    store.close()

    print(f"\nCompleted in {time.time() - wall_start:.1f}s")
    return final_stats

# ─── Plotting (uses sparse in-RAM sample) ─────────────────────────────────────
def generate_plots():
    ts    = np.array(store.s_ts,    dtype=float)
    lids  = np.array(store.s_lids,  dtype=int)
    anom  = np.array(store.s_anom,  dtype=int)
    rssi  = np.array(store.s_rssi,  dtype=float)
    truth = np.array(store.s_truth, dtype=int)

    ts_hr = ts / 3600.0     # convert seconds → hours for axes

    th_times, th_rssi, th_int, th_sim = [], [], [], []
    for (t, r, i, s) in store.th_history:
        th_times.append(t / 3600.0)
        th_rssi.append(r); th_int.append(i); th_sim.append(s)

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(
        f"BLE Beacon Simulation  —  100 hours  |  "
        f"{store.total_count:,} advertisements  |  "
        f"plot sample: 1 in {SAMPLE_EVERY}",
        fontsize=13
    )

    # 1. RSSI over time coloured by ground truth
    ax = axes[0, 0]
    nm = truth == 0;  rg = truth == 1
    ax.plot(ts_hr[nm], rssi[nm], "b.", ms=0.6, alpha=0.25, label="Normal")
    ax.plot(ts_hr[rg], rssi[rg], "r.", ms=1.0, alpha=0.6,  label="Rogue")
    ax.set_xlabel("Simulation time (hours)")
    ax.set_ylabel("RSSI (dBm)")
    ax.set_title("RSSI over Time")
    ax.legend(markerscale=8)

    # 2. Anomaly detections over time
    ax = axes[0, 1]
    ax.plot(ts_hr, anom, "r.", ms=0.6, alpha=0.25)
    ax.set_xlabel("Simulation time (hours)")
    ax.set_ylabel("Anomaly flag (0/1)")
    ax.set_title("Detected Anomalies")

    # 3. Adaptive thresholds
    ax = axes[1, 0]
    if th_times:
        ax.plot(th_times, th_rssi, lw=1.2, label="RSSI threshold")
        ax.plot(th_times, th_int,  lw=1.2, label="Interval threshold")
        ax.plot(th_times, th_sim,  lw=1.2, label="Similarity threshold")
        ax.set_xlabel("Simulation time (hours)")
        ax.set_ylabel("Threshold value")
        ax.set_title("Adaptive Threshold Evolution")
        ax.legend()

    # 4. Top 60 most active logical devices
    ax = axes[1, 1]
    if len(lids) > 0:
        unique, counts = np.unique(lids, return_counts=True)
        idx = np.argsort(counts)[::-1][:60]
        ax.bar(range(len(idx)), counts[idx], color="steelblue")
        ax.set_xlabel("Logical device rank (busiest first)")
        ax.set_ylabel("Sampled advertisement count")
        ax.set_title("Top 60 Logical Devices by Activity")

    # 5. Confusion matrix
    ax = axes[2, 0]
    if len(np.unique(truth)) > 1:
        cm   = confusion_matrix(truth, anom)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=["Normal", "Rogue"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title("Anomaly Detection — Confusion Matrix")
    else:
        ax.set_title("Confusion matrix (need both classes in sample)")

    # 6. Rolling anomaly rate
    ax = axes[2, 1]
    window = min(2000, max(100, len(anom) // 20))
    if len(anom) > window:
        rate = np.convolve(anom.astype(float),
                           np.ones(window) / window, mode="valid")
        ax.plot(ts_hr[window - 1:], rate, "g-", lw=1)
        ax.set_xlabel("Simulation time (hours)")
        ax.set_ylabel("Anomaly rate")
        ax.set_title(f"Rolling Anomaly Rate (window={window} samples)")

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {PLOT_FILE}")

# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    config = {
        "num_static":     50,      # 50 static beacons
        "num_mobile":     30,      # 30 mobile devices
        "rogue_percent":  10.0,    # 10% of devices trigger rogue events
        "duration_hours": 100.0,   # 100-hour simulation
        "width":          500.0,   # 500 x 500 m area
        "height":         500.0,
        "rssi_th":        10.0,
        "int_th":          0.1,
        "sim_th":          0.8,
    }

    total_devices = config["num_static"] + config["num_mobile"]
    sim_sec       = config["duration_hours"] * 3600
    est_ads       = int(total_devices * sim_sec / 0.5)
    est_csv_mb    = est_ads * 120 / 1e6   # ~120 bytes per CSV row

    print("=" * 60)
    print("  Adaptive BLE Beacon Heuristic Engine — 100h Run")
    print("=" * 60)
    print(f"  Estimated advertisements : ~{est_ads:,}")
    print(f"  Estimated CSV size       : ~{est_csv_mb:,.0f} MB")
    print(f"  In-RAM plot sample       : ~{est_ads // SAMPLE_EVERY:,} points")
    print("=" * 60 + "\n")

    final = run_simulation(config)

    print("\nFinal engine stats:")
    print(json.dumps(final, indent=2))
    print(f"\nTotal advertisements : {store.total_count:,}")
    print(f"Rogue advertisements : {store.rogue_count:,}")
    print(f"Anomalies detected   : {store.anomaly_count:,}")
    csv_mb = os.path.getsize(CSV_FILE) / 1e6
    print(f"CSV file size        : {csv_mb:,.1f} MB")

    if store.total_count == 0:
        print("WARNING: no advertisements collected — "
              "check the library compiled correctly.")
    else:
        generate_plots()
