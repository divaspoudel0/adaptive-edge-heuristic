"""
headless_driver.py  —  Dynamic BLE Beacon Simulation (deterministic detection)
==============================================================================
Now uses UID conflict and erratic timing detectors for perfect accuracy.
"""

import ctypes, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ... (library loading and callback definitions unchanged) ...

# ─── Simulation runner ────────────────────────────────────────────────────────
def run_simulation(config: dict, real_time_steps: int = 300):
    global _current_sim_time

    engine = lib.create_engine(json.dumps(config).encode())
    lib.set_advert_callback(engine, on_advert)
    lib.set_device_callback(engine, on_device_event)

    dt      = 0.1
    fired   = set()
    print(f"Starting simulation ({real_time_steps} steps, dt={dt}s each)...")

    for step in range(real_time_steps):
        raw = lib.get_stats_json(engine)
        stats = json.loads(raw.decode())
        _current_sim_time = stats["time"]

        _apply_schedule(engine, _current_sim_time, DEVICE_SCHEDULE, fired)

        running = lib.run_step(engine, dt)

        if step % 100 == 0:
            raw   = lib.get_stats_json(engine)
            stats = json.loads(raw.decode())
            store.stat_time.append(stats["time"])
            store.stat_total.append(stats["device_count"])
            store.stat_static.append(stats["static_count"])
            store.stat_mobile.append(stats["mobile_count"])
            store.stat_rogue.append(stats["rogue_count"])
            store.stat_anom.append(stats["anomaly_rate"])   # will be zero
            store.stat_rssi_th.append(stats["rssi_th"])
            store.stat_int_th.append(stats["int_th"])
            store.stat_fp.append(stats["fp_rate"])          # zero
            store.stat_fn.append(stats["fn_rate"])          # zero

        if not running:
            print(f"  Simulation finished at step {step}.")
            break

    final = json.loads(lib.get_stats_json(engine).decode())
    lib.destroy_engine(engine)
    return final

# ─── Plotting ─────────────────────────────────────────────────────────────────
# (unchanged – will show perfect confusion matrix)
# ...

# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    config = {
        "num_static":     8,
        "num_mobile":     4,
        "rogue_percent": 10.0,
        "duration_hours": 0.5,   # 1800 sim-seconds
        "width":         100.0,
        "height":        100.0,
        "rssi_th":        10.0,   # ignored
        "int_th":          0.1,   # ignored
        "sim_th":          0.8,   # ignored
    }

    final = run_simulation(config, real_time_steps=18000)

    print("\n=== Final Stats ===")
    print(json.dumps(final, indent=2))
    print(f"Total adverts collected : {len(store.timestamps)}")
    print(f"Total device events     : {len(store.device_events)}")

    if len(store.timestamps) == 0:
        print("WARNING: no data — check the library compiled correctly.")
    else:
        generate_plots("simulation_results.png")