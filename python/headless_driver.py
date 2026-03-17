"""
headless_driver.py  —  Dynamic BLE Beacon Simulation
=====================================================
Demonstrates live add/remove of devices while the simulation runs,
then produces an 8-panel plot showing the effect on system behaviour.

Build the library first:
  g++ -O3 -shared -fPIC -std=c++11 \\
      -o cpp/libheuristic.so \\
      cpp/engine.cpp cpp/simulator.cpp cpp/learner.cpp -Icpp

Then run:
  python python/headless_driver.py
"""

import ctypes, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ─── Load library ─────────────────────────────────────────────────────────────
def _load_lib():
    for path in ["./cpp/libheuristic.so",
                 "./cpp/libheuristic.dylib",
                 "./cpp/heuristic.dll"]:
        if os.path.exists(path):
            return ctypes.CDLL(path)
    sys.exit(
        "Library not found. Build with:\n"
        "  g++ -O3 -shared -fPIC -std=c++11 \\\n"
        "      -o cpp/libheuristic.so \\\n"
        "      cpp/engine.cpp cpp/simulator.cpp cpp/learner.cpp -Icpp"
    )

lib = _load_lib()

# ─── ctypes signatures ────────────────────────────────────────────────────────
lib.create_engine.argtypes      = [ctypes.c_char_p]
lib.create_engine.restype       = ctypes.c_void_p
lib.destroy_engine.argtypes     = [ctypes.c_void_p]
lib.destroy_engine.restype      = None

ADVERT_CB = ctypes.CFUNCTYPE(
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

DEVICE_CB = ctypes.CFUNCTYPE(
    None,
    ctypes.c_char_p,   # event: "added" | "removed"
    ctypes.c_char_p,   # device_id
    ctypes.c_char_p,   # device_type: "static" | "mobile" | "rogue"
    ctypes.c_int,      # current total count
)

lib.set_advert_callback.argtypes    = [ctypes.c_void_p, ADVERT_CB]
lib.set_advert_callback.restype     = None
lib.set_device_callback.argtypes    = [ctypes.c_void_p, DEVICE_CB]
lib.set_device_callback.restype     = None
lib.run_step.argtypes               = [ctypes.c_void_p, ctypes.c_double]
lib.run_step.restype                = ctypes.c_int
lib.add_static_devices.argtypes     = [ctypes.c_void_p, ctypes.c_int]
lib.add_static_devices.restype      = ctypes.c_int
lib.add_mobile_devices.argtypes     = [ctypes.c_void_p, ctypes.c_int]
lib.add_mobile_devices.restype      = ctypes.c_int
lib.remove_devices.argtypes         = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
lib.remove_devices.restype          = ctypes.c_int
lib.remove_device_by_id.argtypes    = [ctypes.c_void_p, ctypes.c_char_p]
lib.remove_device_by_id.restype     = ctypes.c_int
lib.get_device_count.argtypes       = [ctypes.c_void_p]
lib.get_device_count.restype        = ctypes.c_int
lib.get_stats_json.argtypes         = [ctypes.c_void_p]
lib.get_stats_json.restype          = ctypes.c_char_p
lib.update_thresholds.argtypes      = [ctypes.c_void_p,
                                        ctypes.c_double,
                                        ctypes.c_double,
                                        ctypes.c_double]
lib.update_thresholds.restype       = None

# ─── Data store ───────────────────────────────────────────────────────────────
class DataStore:
    def __init__(self):
        # Per-advert data
        self.timestamps   = []
        self.rssi_vals    = []
        self.anomalies    = []
        self.rogue_truth  = []
        self.logical_ids  = []

        # Stats snapshots (sampled every 100 steps)
        self.stat_time    = []
        self.stat_total   = []   # total device count
        self.stat_static  = []
        self.stat_mobile  = []
        self.stat_rogue   = []
        self.stat_anom    = []
        self.stat_rssi_th = []
        self.stat_int_th  = []
        self.stat_fp      = []
        self.stat_fn      = []

        # Device-event log  [(sim_time, event, device_type, total)]
        self.device_events = []

store = DataStore()

# ─── Callbacks ────────────────────────────────────────────────────────────────
@ADVERT_CB
def on_advert(ts, mac, uid, svc, rssi, x, y, is_rogue, rtype, lid, anomaly):
    store.timestamps.append(ts)
    store.rssi_vals.append(rssi)
    store.anomalies.append(anomaly)
    store.rogue_truth.append(is_rogue)
    store.logical_ids.append(lid)

# _current_sim_time is updated in the run loop so device events have a timestamp
_current_sim_time = 0.0

@DEVICE_CB
def on_device_event(event, device_id, device_type, total):
    store.device_events.append((
        _current_sim_time,
        event.decode(),
        device_type.decode(),
        total,
    ))

# ─── Device schedule ──────────────────────────────────────────────────────────
# Each entry: (sim_time_seconds, action, arg)
#   action = "add_static"  → arg = count
#   action = "add_mobile"  → arg = count
#   action = "remove"      → arg = count  (removes mobile/rogue only)
#   action = "remove_rogue"→ arg = count  (removes rogue only)
#
# Times are in simulation seconds. With duration_hours=0.5 the sim runs 1800s.
# We demonstrate three phases:
#   Phase 1 (0–600s):   initial fleet
#   Phase 2 (600s):     scale-up event — venue opens, many devices join
#   Phase 3 (900s):     partial scale-down — some devices leave
#   Phase 4 (1200s):    second wave of mobile devices
#   Phase 5 (1500s):    clean-up — remove excess mobiles

DEVICE_SCHEDULE = [
    (600.0,  "add_static",   5),   # 5 new beacons installed mid-run
    (600.0,  "add_mobile",  10),   # crowd arrives
    (900.0,  "remove",       8),   # crowd thins
    (1200.0, "add_mobile",  15),   # second wave
    (1500.0, "remove",      10),   # end of event
]

def _apply_schedule(engine, sim_time, schedule, fired):
    """Fire any scheduled actions whose time has passed."""
    for idx, (t, action, arg) in enumerate(schedule):
        if idx in fired:
            continue
        if sim_time >= t:
            if   action == "add_static":
                new_total = lib.add_static_devices(engine, arg)
                print(f"  [t={sim_time:.0f}s] +{arg} static beacons → total={new_total}")
            elif action == "add_mobile":
                new_total = lib.add_mobile_devices(engine, arg)
                print(f"  [t={sim_time:.0f}s] +{arg} mobile devices → total={new_total}")
            elif action == "remove":
                new_total = lib.remove_devices(engine, arg, 0)
                print(f"  [t={sim_time:.0f}s] -{arg} devices removed → total={new_total}")
            elif action == "remove_rogue":
                new_total = lib.remove_devices(engine, arg, 1)
                print(f"  [t={sim_time:.0f}s] -{arg} rogues removed → total={new_total}")
            fired.add(idx)

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
        # Get current sim time from stats before stepping
        raw = lib.get_stats_json(engine)
        stats = json.loads(raw.decode())
        _current_sim_time = stats["time"]

        # Apply scheduled dynamic changes
        _apply_schedule(engine, _current_sim_time, DEVICE_SCHEDULE, fired)

        # Run one simulation step
        running = lib.run_step(engine, dt)

        # Collect stats snapshot every 100 steps
        if step % 100 == 0:
            raw   = lib.get_stats_json(engine)
            stats = json.loads(raw.decode())
            store.stat_time.append(stats["time"])
            store.stat_total.append(stats["device_count"])
            store.stat_static.append(stats["static_count"])
            store.stat_mobile.append(stats["mobile_count"])
            store.stat_rogue.append(stats["rogue_count"])
            store.stat_anom.append(stats["anomaly_rate"])
            store.stat_rssi_th.append(stats["rssi_th"])
            store.stat_int_th.append(stats["int_th"])
            store.stat_fp.append(stats["fp_rate"])
            store.stat_fn.append(stats["fn_rate"])

        if not running:
            print(f"  Simulation finished at step {step}.")
            break

    final = json.loads(lib.get_stats_json(engine).decode())
    lib.destroy_engine(engine)
    return final

# ─── Plotting ─────────────────────────────────────────────────────────────────
def generate_plots(outfile="simulation_results.png"):
    ts    = np.array(store.timestamps,  dtype=float)
    rssi  = np.array(store.rssi_vals,   dtype=float)
    anom  = np.array(store.anomalies,   dtype=int)
    truth = np.array(store.rogue_truth, dtype=int)
    lids  = np.array(store.logical_ids, dtype=int)

    st    = np.array(store.stat_time,    dtype=float)
    s_tot = np.array(store.stat_total,   dtype=int)
    s_sta = np.array(store.stat_static,  dtype=int)
    s_mob = np.array(store.stat_mobile,  dtype=int)
    s_rog = np.array(store.stat_rogue,   dtype=int)
    s_anom= np.array(store.stat_anom,    dtype=float)
    s_rssi= np.array(store.stat_rssi_th, dtype=float)
    s_int = np.array(store.stat_int_th,  dtype=float)
    s_fp  = np.array(store.stat_fp,      dtype=float)
    s_fn  = np.array(store.stat_fn,      dtype=float)

    # Schedule annotation helper
    sched_times = [t for (t, _, _) in DEVICE_SCHEDULE]
    sched_labels = [
        "+5 static", "+10 mobile", "-8 devices",
        "+15 mobile", "-10 devices",
    ]

    def add_schedule_lines(ax, color="gray", alpha=0.35):
        for t, label in zip(sched_times, sched_labels):
            ax.axvline(t, color=color, linestyle="--", linewidth=0.8, alpha=alpha)

    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    fig.suptitle(
        "Dynamic BLE Beacon Heuristic Engine — Simulation Results",
        fontsize=15, fontweight="bold", y=0.98
    )

    # ── 1. Device count over time (stacked area) ──────────────────────────────
    ax = axes[0, 0]
    if len(st) > 0:
        ax.stackplot(st,
                     s_sta, s_mob, s_rog,
                     labels=["Static beacons", "Mobile devices", "Rogues"],
                     colors=["#4e79a7", "#f28e2b", "#e15759"],
                     alpha=0.75)
        add_schedule_lines(ax)
        for t, lbl in zip(sched_times, sched_labels):
            ax.annotate(lbl, xy=(t, 0), xytext=(t + 5, ax.get_ylim()[1] * 0.9),
                        fontsize=7, color="gray",
                        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))
        ax.set_xlabel("Simulation time (s)")
        ax.set_ylabel("Device count")
        ax.set_title("Live Device Count (stacked by type)")
        ax.legend(loc="upper left", fontsize=8)
    else:
        ax.set_title("Device count (no data)")

    # ── 2. RSSI over time coloured by ground truth ────────────────────────────
    ax = axes[0, 1]
    nm = truth == 0; rg = truth == 1
    if nm.any():
        ax.plot(ts[nm], rssi[nm], "b.", markersize=1, alpha=0.3, label="Normal")
    if rg.any():
        ax.plot(ts[rg], rssi[rg], "r.", markersize=2, alpha=0.7, label="Rogue")
    add_schedule_lines(ax)
    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("RSSI (dBm)")
    ax.set_title("RSSI over Time")
    ax.legend(markerscale=4, fontsize=8)

    # ── 3. Anomaly detection rate vs device count ─────────────────────────────
    ax = axes[1, 0]
    if len(st) > 0:
        color_anom = "#e15759"
        ax.plot(st, s_anom * 100, color=color_anom, linewidth=1.5,
                label="Anomaly rate %")
        add_schedule_lines(ax)
        ax2 = ax.twinx()
        ax2.plot(st, s_tot, color="#4e79a7", linewidth=1, linestyle=":",
                 label="Total devices")
        ax2.set_ylabel("Total device count", color="#4e79a7")
        ax2.tick_params(axis="y", labelcolor="#4e79a7")
        ax.set_xlabel("Simulation time (s)")
        ax.set_ylabel("Anomaly rate (%)", color=color_anom)
        ax.tick_params(axis="y", labelcolor=color_anom)
        ax.set_title("Anomaly Rate vs Device Count")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    else:
        ax.set_title("Anomaly rate (no data)")

    # ── 4. Adaptive thresholds over time ─────────────────────────────────────
    ax = axes[1, 1]
    if len(st) > 0:
        ax.plot(st, s_rssi, label="RSSI threshold",     linewidth=1.5)
        ax.plot(st, s_int,  label="Interval threshold", linewidth=1.5)
        add_schedule_lines(ax)
        ax.set_xlabel("Simulation time (s)")
        ax.set_ylabel("Threshold value")
        ax.set_title("Adaptive Thresholds (self-adjusting)")
        ax.legend(fontsize=8)
    else:
        ax.set_title("Adaptive thresholds (no data)")

    # ── 5. FP / FN rates over time ────────────────────────────────────────────
    ax = axes[2, 0]
    if len(st) > 0:
        ax.plot(st, s_fp * 100, label="False positive rate %",
                color="orange", linewidth=1.5)
        ax.plot(st, s_fn * 100, label="False negative rate %",
                color="red",    linewidth=1.5, linestyle="--")
        add_schedule_lines(ax)
        ax.set_xlabel("Simulation time (s)")
        ax.set_ylabel("Error rate (%)")
        ax.set_title("False Positive / Negative Rates")
        ax.legend(fontsize=8)
    else:
        ax.set_title("FP/FN rates (no data)")

    # ── 6. Confusion matrix ───────────────────────────────────────────────────
    ax = axes[2, 1]
    if len(truth) > 0 and len(np.unique(truth)) > 1:
        cm   = confusion_matrix(truth, anom)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=["Normal", "Rogue"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title("Anomaly Detection — Confusion Matrix")
    else:
        ax.set_title("Confusion matrix (need both classes)")

    # ── 7. Advertisements per logical device ID ───────────────────────────────
    ax = axes[3, 0]
    if len(lids) > 0:
        unique, counts = np.unique(lids, return_counts=True)
        ax.bar(unique, counts, color="#4e79a7", alpha=0.8)
        ax.set_xlabel("Logical device ID")
        ax.set_ylabel("Advertisement count")
        ax.set_title("Advertisements per Logical Device")
    else:
        ax.set_title("Logical device adverts (no data)")

    # ── 8. Device event timeline ──────────────────────────────────────────────
    ax = axes[3, 1]
    if store.device_events:
        ev_times  = [e[0] for e in store.device_events]
        ev_types  = [e[2] for e in store.device_events]
        ev_events = [e[1] for e in store.device_events]
        colors    = {"static": "#4e79a7", "mobile": "#f28e2b", "rogue": "#e15759"}
        y_vals    = {"added": 1, "removed": -1}
        for t, dtype, ev in zip(ev_times, ev_types, ev_events):
            col = colors.get(dtype, "gray")
            y   = y_vals.get(ev, 0)
            ax.scatter(t, y, color=col, s=10, alpha=0.6)
        add_schedule_lines(ax)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_yticks([1, -1])
        ax.set_yticklabels(["Added", "Removed"])
        ax.set_xlabel("Simulation time (s)")
        ax.set_title("Device Event Timeline")
        patches = [mpatches.Patch(color=c, label=t)
                   for t, c in colors.items()]
        ax.legend(handles=patches, fontsize=8)
    else:
        ax.set_title("Device event timeline (no events)")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {outfile}")

# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    config = {
        "num_static":     8,
        "num_mobile":     4,
        "rogue_percent": 10.0,
        "duration_hours": 0.5,   # 1800 sim-seconds
        "width":         100.0,
        "height":        100.0,
        "rssi_th":        10.0,
        "int_th":          0.1,
        "sim_th":          0.8,
    }

    # 18000 steps × dt=0.1s = 1800 sim-seconds (covers full schedule)
    final = run_simulation(config, real_time_steps=18000)

    print("\n=== Final Stats ===")
    print(json.dumps(final, indent=2))
    print(f"Total adverts collected : {len(store.timestamps)}")
    print(f"Total device events     : {len(store.device_events)}")

    if len(store.timestamps) == 0:
        print("WARNING: no data — check the library compiled correctly.")
    else:
        generate_plots("simulation_results.png")
