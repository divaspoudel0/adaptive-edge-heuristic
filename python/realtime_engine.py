#!/usr/bin/env python3
"""
realtime_engine.py  —  Interactive BLE Beacon Monitor (Kathmandu)
=================================================================
Runs indefinitely until you press Ctrl+C or type 'q'.

Commands (type while running):
  1+        add 1 static beacon
  1-        remove 1 static beacon
  2+        add 1 mobile device
  2-        remove 1 mobile device
  3+        inject 1 rogue device (spoof_uid)
  3-        remove oldest rogue
  1+5       add 5 static beacons
  2-3       remove 3 mobile devices
  3+e       inject erratic_timing rogue
  3+r       inject replay rogue
  s         print current stats
  h         help
  q / quit  stop simulation and generate plots

Data is saved to  data/  and merged across sessions.
"""

import ctypes, csv, json, os, queue, signal, sys, time, threading
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

try:
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ══════════════════════════════════════════════════════════════════════════════
# Kathmandu — Thamel district (realistic 500 × 500 m BLE deployment zone)
# ══════════════════════════════════════════════════════════════════════════════
KTM_LAT_SW  = 27.7100     # southern boundary
KTM_LON_SW  = 85.3070     # western boundary
KTM_LAT_NE  = 27.7155     # northern boundary
KTM_LON_NE  = 85.3128     # eastern boundary
SIM_WIDTH   = 500.0        # metres
SIM_HEIGHT  = 500.0        # metres

# Notable Thamel landmarks (used in display)
LANDMARKS = {
    "Thamel Chowk":       (0.50, 0.50),
    "Garden of Dreams":   (0.82, 0.45),
    "Kathmandu Mall":     (0.30, 0.20),
    "Rani Pokhari":       (0.90, 0.10),
    "Durbar Marg":        (0.70, 0.05),
}

def xy_to_latlon(x: float, y: float):
    lat = KTM_LAT_SW + (y / SIM_HEIGHT) * (KTM_LAT_NE - KTM_LAT_SW)
    lon = KTM_LON_SW + (x / SIM_WIDTH)  * (KTM_LON_NE - KTM_LON_SW)
    return round(lat, 6), round(lon, 6)

def nearest_landmark(x: float, y: float) -> str:
    nx, ny = x / SIM_WIDTH, y / SIM_HEIGHT
    best, best_d = "", 1e9
    for name, (lx, ly) in LANDMARKS.items():
        d = ((nx - lx)**2 + (ny - ly)**2) ** 0.5
        if d < best_d:
            best_d, best = d, name
    return best

# ══════════════════════════════════════════════════════════════════════════════
# Library loading & ctypes setup
# ══════════════════════════════════════════════════════════════════════════════
def _load_lib():
    for path in ["./cpp/libheuristic.so",
                 "./cpp/libheuristic.dylib",
                 "./cpp/heuristic.dll"]:
        if os.path.exists(path):
            return ctypes.CDLL(path)
    sys.exit(
        "\nLibrary not found. Build with:\n"
        "  g++ -O3 -shared -fPIC -std=c++11 \\\n"
        "      -o cpp/libheuristic.so \\\n"
        "      cpp/engine.cpp cpp/simulator.cpp cpp/learner.cpp -Icpp\n"
    )

_lib = _load_lib()

ADVERT_CB = ctypes.CFUNCTYPE(
    None,
    ctypes.c_double, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
    ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.c_int,    ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
)
DEVICE_CB = ctypes.CFUNCTYPE(
    None,
    ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int,
)

for name, argtypes, restype in [
    ("create_engine",       [ctypes.c_char_p],                           ctypes.c_void_p),
    ("destroy_engine",      [ctypes.c_void_p],                           None),
    ("set_advert_callback", [ctypes.c_void_p, ADVERT_CB],                None),
    ("set_device_callback", [ctypes.c_void_p, DEVICE_CB],                None),
    ("run_step",            [ctypes.c_void_p, ctypes.c_double],          ctypes.c_int),
    ("add_static_devices",  [ctypes.c_void_p, ctypes.c_int],             ctypes.c_int),
    ("add_mobile_devices",  [ctypes.c_void_p, ctypes.c_int],             ctypes.c_int),
    ("remove_devices",      [ctypes.c_void_p, ctypes.c_int, ctypes.c_int], ctypes.c_int),
    ("inject_rogue_now",    [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_double], ctypes.c_int),
    ("get_device_count",    [ctypes.c_void_p],                           ctypes.c_int),
    ("get_static_count",    [ctypes.c_void_p],                           ctypes.c_int),
    ("get_mobile_count",    [ctypes.c_void_p],                           ctypes.c_int),
    ("get_rogue_count",     [ctypes.c_void_p],                           ctypes.c_int),
    ("get_stats_json",      [ctypes.c_void_p],                           ctypes.c_char_p),
    ("update_thresholds",   [ctypes.c_void_p, ctypes.c_double,
                              ctypes.c_double, ctypes.c_double],         None),
]:
    fn = getattr(_lib, name)
    fn.argtypes = argtypes
    fn.restype  = restype

# ══════════════════════════════════════════════════════════════════════════════
# Data persistence
# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR    = Path("data")
ADVERTS_CSV = DATA_DIR / "adverts.csv"
EVENTS_CSV  = DATA_DIR / "events.csv"
SESSIONS_JS = DATA_DIR / "sessions.json"

ADVERT_FIELDS = [
    "session_id", "timestamp", "mac", "uid", "service_id",
    "rssi", "x", "y", "lat", "lon", "landmark",
    "is_rogue", "rogue_type", "logical_id", "anomaly",
]
EVENT_FIELDS = [
    "session_id", "sim_time", "real_time",
    "event", "device_id", "device_type", "total_count",
]

class DataManager:
    def __init__(self, session_id: str):
        DATA_DIR.mkdir(exist_ok=True)
        self.session_id = session_id
        self._advert_buf: list = []
        self._event_buf:  list = []
        self._buf_lock = threading.Lock()
        self._flush_every = 500   # rows before auto-flush

        # Ensure CSV headers exist
        self._ensure_header(ADVERTS_CSV, ADVERT_FIELDS)
        self._ensure_header(EVENTS_CSV,  EVENT_FIELDS)

    def _ensure_header(self, path: Path, fields: list):
        if not path.exists():
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(fields)

    def record_advert(self, row: dict):
        with self._buf_lock:
            self._advert_buf.append(row)
            if len(self._advert_buf) >= self._flush_every:
                self._flush_adverts()

    def record_event(self, row: dict):
        with self._buf_lock:
            self._event_buf.append(row)
            if len(self._event_buf) >= 100:
                self._flush_events()

    def _flush_adverts(self):
        if not self._advert_buf:
            return
        with open(ADVERTS_CSV, "a", newline="") as f:
            w = csv.DictWriter(f, ADVERT_FIELDS, extrasaction="ignore")
            w.writerows(self._advert_buf)
        self._advert_buf.clear()

    def _flush_events(self):
        if not self._event_buf:
            return
        with open(EVENTS_CSV, "a", newline="") as f:
            w = csv.DictWriter(f, EVENT_FIELDS, extrasaction="ignore")
            w.writerows(self._event_buf)
        self._event_buf.clear()

    def flush_all(self):
        with self._buf_lock:
            self._flush_adverts()
            self._flush_events()

    def save_session_meta(self, meta: dict):
        sessions = []
        if SESSIONS_JS.exists():
            try:
                sessions = json.loads(SESSIONS_JS.read_text())
            except Exception:
                sessions = []
        sessions.append(meta)
        SESSIONS_JS.write_text(json.dumps(sessions, indent=2))

    @staticmethod
    def load_all_adverts() -> list:
        if not ADVERTS_CSV.exists():
            return []
        with open(ADVERTS_CSV, newline="") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def load_all_events() -> list:
        if not EVENTS_CSV.exists():
            return []
        with open(EVENTS_CSV, newline="") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def load_sessions() -> list:
        if not SESSIONS_JS.exists():
            return []
        try:
            return json.loads(SESSIONS_JS.read_text())
        except Exception:
            return []

# ══════════════════════════════════════════════════════════════════════════════
# Shared state (engine lock + live stats)
# ══════════════════════════════════════════════════════════════════════════════
class SharedState:
    def __init__(self):
        self.lock        = threading.Lock()
        self.stats       = {}
        self.rogue_alerts: list = []    # last 5 rogue detections for display
        self.total_adverts   = 0
        self.advert_rate     = 0.0      # rolling adverts/s
        self._rate_window: list = []    # (real_time, count) for rate calc
        self.running     = True

state = SharedState()

# ══════════════════════════════════════════════════════════════════════════════
# Command parser
# ══════════════════════════════════════════════════════════════════════════════
ROGUE_SUBTYPES = {"s": "spoof_uid", "e": "erratic_timing", "r": "replay"}

def parse_command(raw: str):
    """
    Returns (action, type_num, count, extra) or None if invalid.
    Examples:
      "1+"    → ("add",    1, 1,  None)
      "2-3"   → ("remove", 2, 3,  None)
      "3+e"   → ("add",    3, 1, "erratic_timing")
      "3-"    → ("remove", 3, 1,  None)
      "1+10"  → ("add",    1, 10, None)
    """
    raw = raw.strip()
    if not raw:
        return None
    if raw in ("q", "quit", "exit"):
        return ("quit", 0, 0, None)
    if raw in ("s", "stats"):
        return ("stats", 0, 0, None)
    if raw in ("h", "help"):
        return ("help", 0, 0, None)

    # Expect: digit (+|-) [digit|letter]
    if len(raw) < 2:
        return None
    try:
        dev_type = int(raw[0])
    except ValueError:
        return None
    if dev_type not in (1, 2, 3):
        return None
    op = raw[1]
    if op not in ("+", "-"):
        return None

    suffix = raw[2:].strip() if len(raw) > 2 else ""

    # Rogue sub-type letter
    rogue_subtype = None
    count = 1
    if dev_type == 3 and op == "+" and suffix in ROGUE_SUBTYPES:
        rogue_subtype = ROGUE_SUBTYPES[suffix]
    elif suffix.isdigit():
        count = max(1, int(suffix))
    elif suffix:
        return None  # unrecognised suffix

    action = "add" if op == "+" else "remove"
    return (action, dev_type, count, rogue_subtype)

# ══════════════════════════════════════════════════════════════════════════════
# Display (writes to stdout from a background thread)
# ══════════════════════════════════════════════════════════════════════════════
ANSI_CLEAR  = "\033[2J\033[H"
ANSI_BOLD   = "\033[1m"
ANSI_RED    = "\033[91m"
ANSI_YELLOW = "\033[93m"
ANSI_GREEN  = "\033[92m"
ANSI_CYAN   = "\033[96m"
ANSI_RESET  = "\033[0m"
ANSI_DIM    = "\033[2m"
BAR         = "─" * 62

def fmt_time(seconds: float) -> str:
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    return f"{h:02d}h {m:02d}m {s:02d}s"

def render_status(start_real: float):
    with state.lock:
        st = dict(state.stats)
        alerts = list(state.rogue_alerts)
        total  = state.total_adverts
        rate   = state.advert_rate

    elapsed = time.time() - start_real
    sim_t   = st.get("time", 0.0)
    n_dev   = st.get("device_count",  0)
    n_sta   = st.get("static_count",  0)
    n_mob   = st.get("mobile_count",  0)
    n_rog   = st.get("rogue_count",   0)
    a_rate  = st.get("anomaly_rate",  0.0)
    fp      = st.get("fp_rate",       0.0)
    fn      = st.get("fn_rate",       0.0)
    rssi_th = st.get("rssi_th",      10.0)
    int_th  = st.get("int_th",        0.1)

    rogue_indicator = (ANSI_RED + " ⚠ ROGUES ACTIVE" + ANSI_RESET) if n_rog > 0 else (ANSI_GREEN + " ✓ CLEAN" + ANSI_RESET)

    lines = [
        f"\n{ANSI_BOLD}{BAR}{ANSI_RESET}",
        f"{ANSI_BOLD}{ANSI_CYAN}  BLE Beacon Monitor │ Kathmandu (Thamel){ANSI_RESET}",
        BAR,
        f"  Real elapsed : {fmt_time(elapsed)}   Sim time : {fmt_time(sim_t)}",
        f"  Devices  : {ANSI_BOLD}Static={n_sta}  Mobile={n_mob}  Rogues={n_rog}  Total={n_dev}{ANSI_RESET}{rogue_indicator}",
        f"  Adverts  : {ANSI_BOLD}{total:,}{ANSI_RESET}  ({rate:.1f}/s)",
        f"  Anomaly  : {a_rate*100:.1f}%   FP={fp*100:.1f}%   FN={fn*100:.1f}%",
        f"  Thresholds: RSSI={rssi_th:.2f}  Interval={int_th:.3f}",
        BAR,
    ]

    if alerts:
        lines.append(f"  {ANSI_RED}{ANSI_BOLD}Recent rogue detections:{ANSI_RESET}")
        for a in alerts[-3:]:
            lines.append(f"   {ANSI_RED}⚠{ANSI_RESET}  {a}")
        lines.append(BAR)

    lines += [
        f"  {ANSI_DIM}Commands: TYPE+[N]  TYPE-[N]   (1=static 2=mobile 3=rogue){ANSI_RESET}",
        f"  {ANSI_DIM}  3+s=spoof_uid  3+e=erratic  3+r=replay  |  q=quit  s=stats{ANSI_RESET}",
        BAR,
        f"\n> ",
    ]
    sys.stdout.write("\r" + "\n".join(lines))
    sys.stdout.flush()

# ══════════════════════════════════════════════════════════════════════════════
# Simulation thread
# ══════════════════════════════════════════════════════════════════════════════
class SimulationThread(threading.Thread):
    def __init__(self, engine, data_mgr: DataManager,
                 cmd_queue: queue.Queue, session_id: str):
        super().__init__(daemon=True)
        self.engine     = engine
        self.data_mgr   = data_mgr
        self.cmd_queue  = cmd_queue
        self.session_id = session_id
        self.dt         = 0.1       # sim seconds per step
        self._step_count = 0
        self._real_start = time.time()

    def run(self):
        while state.running:
            # ── Process pending commands ────────────────────────────────────
            while not self.cmd_queue.empty():
                try:
                    cmd = self.cmd_queue.get_nowait()
                    self._apply_command(cmd)
                except queue.Empty:
                    break

            # ── Single simulation step ──────────────────────────────────────
            with state.lock:
                _lib.run_step(self.engine, self.dt)

            self._step_count += 1

            # ── Refresh stats every 200 steps ──────────────────────────────
            if self._step_count % 200 == 0:
                raw   = _lib.get_stats_json(self.engine)
                stats = json.loads(raw.decode())
                elapsed = time.time() - self._real_start
                with state.lock:
                    state.stats = stats
                    # Rolling advert rate (adverts per real second)
                    now = time.time()
                    state._rate_window.append((now, state.total_adverts))
                    if len(state._rate_window) > 20:
                        state._rate_window.pop(0)
                    if len(state._rate_window) >= 2:
                        dt_w = state._rate_window[-1][0] - state._rate_window[0][0]
                        dn_w = state._rate_window[-1][1] - state._rate_window[0][1]
                        state.advert_rate = dn_w / dt_w if dt_w > 0 else 0.0

            # Small yield to keep input thread responsive
            time.sleep(0.0005)

    def _apply_command(self, parsed):
        action, dev_type, count, extra = parsed
        with state.lock:
            if action == "add":
                if dev_type == 1:
                    n = _lib.add_static_devices(self.engine, count)
                    self._alert(f"Added {count} static beacon(s) → total={n}")
                elif dev_type == 2:
                    n = _lib.add_mobile_devices(self.engine, count)
                    self._alert(f"Added {count} mobile device(s) → total={n}")
                elif dev_type == 3:
                    rtype = extra if extra else "spoof_uid"
                    for _ in range(count):
                        n = _lib.inject_rogue_now(
                            self.engine,
                            rtype.encode(),
                            ctypes.c_double(300.0)
                        )
                    self._alert(f"Injected rogue [{rtype}] → total={n}")
            elif action == "remove":
                if dev_type == 1:
                    # Remove static by removing any non-mobile, non-rogue
                    n = _lib.remove_devices(self.engine, count, 0)
                    self._alert(f"Removed {count} device(s) → total={n}")
                elif dev_type == 2:
                    n = _lib.remove_devices(self.engine, count, 0)
                    self._alert(f"Removed {count} mobile device(s) → total={n}")
                elif dev_type == 3:
                    n = _lib.remove_devices(self.engine, count, 1)
                    self._alert(f"Removed {count} rogue(s) → total={n}")

    def _alert(self, msg: str):
        # state.lock already held by caller
        state.rogue_alerts.append(
            f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        )
        if len(state.rogue_alerts) > 20:
            state.rogue_alerts.pop(0)

# ══════════════════════════════════════════════════════════════════════════════
# Callbacks (called from sim thread — must be fast, no heavy I/O)
# ══════════════════════════════════════════════════════════════════════════════
_data_mgr_ref = None   # set in main()
_session_id_ref = ""

@ADVERT_CB
def on_advert(ts, mac, uid, svc, rssi, x, y, is_rogue, rtype, lid, anomaly):
    lat, lon = xy_to_latlon(x, y)
    lm       = nearest_landmark(x, y)
    rt       = rtype.decode() if rtype else ""

    with state.lock:
        state.total_adverts += 1

    # Rogue alert (anomaly + known ground truth rogue)
    if is_rogue and anomaly:
        alert = (f"[{datetime.now().strftime('%H:%M:%S')}] "
                 f"{ANSI_BOLD}{rt}{ANSI_RESET} detected near "
                 f"{ANSI_BOLD}{lm}{ANSI_RESET} "
                 f"({lat}°N, {lon}°E)  RSSI={rssi:.1f}dBm")
        with state.lock:
            state.rogue_alerts.append(alert)
            if len(state.rogue_alerts) > 20:
                state.rogue_alerts.pop(0)

    if _data_mgr_ref:
        _data_mgr_ref.record_advert({
            "session_id": _session_id_ref,
            "timestamp":  round(ts, 3),
            "mac":        mac.decode(),
            "uid":        uid.decode(),
            "service_id": svc.decode(),
            "rssi":       round(rssi, 2),
            "x":          round(x, 2),
            "y":          round(y, 2),
            "lat":        lat,
            "lon":        lon,
            "landmark":   lm,
            "is_rogue":   is_rogue,
            "rogue_type": rt,
            "logical_id": lid,
            "anomaly":    anomaly,
        })

@DEVICE_CB
def on_device_event(event, device_id, device_type, total):
    if _data_mgr_ref:
        _data_mgr_ref.record_event({
            "session_id":  _session_id_ref,
            "sim_time":    state.stats.get("time", 0),
            "real_time":   datetime.now().isoformat(),
            "event":       event.decode(),
            "device_id":   device_id.decode(),
            "device_type": device_type.decode(),
            "total_count": total,
        })

# ══════════════════════════════════════════════════════════════════════════════
# Plotting — runs after simulation stops, uses ALL accumulated CSV data
# ══════════════════════════════════════════════════════════════════════════════
def generate_plots(session_id: str):
    print("\nLoading accumulated data for plotting…")
    rows   = DataManager.load_all_adverts()
    events = DataManager.load_all_events()
    sessions_meta = DataManager.load_sessions()

    if not rows:
        print("No data to plot.")
        return

    # Convert to numpy
    def col(r, k, dtype=float):
        return np.array([dtype(x[k]) for x in r if k in x], dtype=dtype if dtype != str else object)

    ts      = col(rows, "timestamp")
    rssi    = col(rows, "rssi")
    anomaly = col(rows, "anomaly", int)
    truth   = col(rows, "is_rogue", int)
    lids    = col(rows, "logical_id", int)
    lat_arr = col(rows, "lat")
    lon_arr = col(rows, "lon")
    sess_arr= np.array([r["session_id"] for r in rows])

    # Stats snapshots — reconstruct from device events
    ev_times  = [float(e["sim_time"]) for e in events]
    ev_types  = [e["device_type"] for e in events]
    ev_events = [e["event"]       for e in events]
    ev_sess   = [e["session_id"]  for e in events]

    # Unique session list for colour coding
    unique_sessions = sorted(set(sess_arr))
    cmap = plt.get_cmap("tab10")
    sess_colors = {s: cmap(i % 10) for i, s in enumerate(unique_sessions)}

    out = f"simulation_results_{session_id}.png"
    fig, axes = plt.subplots(4, 2, figsize=(18, 20))
    fig.suptitle(
        f"BLE Beacon Monitor — Kathmandu (Thamel)\n"
        f"Sessions: {len(unique_sessions)}  |  Total adverts: {len(rows):,}",
        fontsize=14, fontweight="bold"
    )

    # ── 1. RSSI over time (coloured by session) ───────────────────────────────
    ax = axes[0, 0]
    for s in unique_sessions:
        mask = sess_arr == s
        nm   = mask & (truth == 0)
        rg   = mask & (truth == 1)
        c    = sess_colors[s]
        if nm.any(): ax.plot(ts[nm], rssi[nm], ".", color=c, markersize=1, alpha=0.25)
        if rg.any(): ax.plot(ts[rg], rssi[rg], "x", color="red", markersize=2, alpha=0.6)
    ax.set_xlabel("Sim time (s)"); ax.set_ylabel("RSSI (dBm)")
    ax.set_title("RSSI over Time  (red × = ground-truth rogue)")

    # ── 2. GPS scatter map of Kathmandu (Thamel) ─────────────────────────────
    ax = axes[0, 1]
    nm_mask = truth == 0; rg_mask = truth == 1; an_mask = anomaly == 1
    if nm_mask.any():
        ax.scatter(lon_arr[nm_mask], lat_arr[nm_mask],
                   s=2, c="steelblue", alpha=0.15, label="Normal")
    if rg_mask.any():
        ax.scatter(lon_arr[rg_mask], lat_arr[rg_mask],
                   s=6, c="red", alpha=0.5, label="Rogue (truth)")
    detected = rg_mask & an_mask
    if detected.any():
        ax.scatter(lon_arr[detected], lat_arr[detected],
                   s=20, c="orange", marker="*", alpha=0.8, label="Detected")
    # Landmark annotations
    for name, (rx, ry) in LANDMARKS.items():
        plat = KTM_LAT_SW + ry * (KTM_LAT_NE - KTM_LAT_SW)
        plon = KTM_LON_SW + rx * (KTM_LON_NE - KTM_LON_SW)
        ax.annotate(name, (plon, plat), fontsize=6, color="gray",
                    ha="center", va="bottom")
        ax.plot(plon, plat, "k+", markersize=5)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Device Location Map — Thamel, Kathmandu")
    ax.legend(markerscale=4, fontsize=7)
    ax.set_xlim(KTM_LON_SW, KTM_LON_NE)
    ax.set_ylim(KTM_LAT_SW, KTM_LAT_NE)

    # ── 3. Anomaly rate over simulation time (rolling) ────────────────────────
    ax = axes[1, 0]
    window = max(500, len(anomaly) // 50)
    if len(anomaly) > window:
        rate = np.convolve(anomaly.astype(float), np.ones(window)/window, mode="valid")
        ax.plot(ts[window-1:], rate * 100, color="#e15759", linewidth=1.2)
    # Session boundary lines
    for s in unique_sessions[1:]:
        mask = sess_arr == s
        if mask.any():
            ax.axvline(ts[mask][0], color="gray", linestyle="--",
                       linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Sim time (s)"); ax.set_ylabel("Anomaly rate (%)")
    ax.set_title(f"Rolling Anomaly Rate (window={window})")

    # ── 4. Device event timeline ──────────────────────────────────────────────
    ax = axes[1, 1]
    type_colors = {"static": "#4e79a7", "mobile": "#f28e2b", "rogue": "#e15759"}
    if ev_times:
        for t, dtype, ev in zip(ev_times, ev_types, ev_events):
            y = 1 if ev == "added" else -1
            ax.scatter(t, y, color=type_colors.get(dtype, "gray"), s=8, alpha=0.5)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_yticks([1, -1]); ax.set_yticklabels(["Added", "Removed"])
        ax.set_xlabel("Sim time (s)"); ax.set_title("Device Event Timeline")
        patches = [mpatches.Patch(color=c, label=t) for t, c in type_colors.items()]
        ax.legend(handles=patches, fontsize=8)
    else:
        ax.set_title("Device Event Timeline (no events)")

    # ── 5. Confusion matrix ───────────────────────────────────────────────────
    ax = axes[2, 0]
    if HAS_SKLEARN and len(np.unique(truth)) > 1:
        cm   = confusion_matrix(truth, anomaly)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Rogue"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        tn, fp, fn, tp = cm.ravel()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
        ax.set_title(f"Confusion Matrix  P={prec:.2f}  R={rec:.2f}  F1={f1:.2f}")
    else:
        ax.set_title("Confusion Matrix (need sklearn + both classes)")

    # ── 6. Advertisements per logical device ──────────────────────────────────
    ax = axes[2, 1]
    if len(lids) > 0:
        unique_lids, counts = np.unique(lids[lids >= 0], return_counts=True)
        top_n = min(40, len(unique_lids))
        idx   = np.argsort(counts)[-top_n:][::-1]
        ax.bar(range(top_n), counts[idx], color="#4e79a7", alpha=0.8)
        ax.set_xlabel("Logical device rank"); ax.set_ylabel("Advert count")
        ax.set_title(f"Top {top_n} Logical Devices by Advert Count")
    else:
        ax.set_title("Logical devices (no data)")

    # ── 7. Session summary table ──────────────────────────────────────────────
    ax = axes[3, 0]
    ax.axis("off")
    if sessions_meta:
        headers = ["Session", "Start", "Adverts", "Duration"]
        cells   = []
        for sm in sessions_meta[-8:]:  # last 8 sessions
            cells.append([
                sm.get("session_id", "")[:12],
                sm.get("start_time", "")[:19],
                f"{sm.get('total_adverts', 0):,}",
                sm.get("duration", ""),
            ])
        tbl = ax.table(cellText=cells, colLabels=headers,
                       loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.4)
        ax.set_title("Session History", pad=20)
    else:
        ax.text(0.5, 0.5, "No session history yet",
                ha="center", va="center", transform=ax.transAxes)

    # ── 8. Rogue detection by type ────────────────────────────────────────────
    ax = axes[3, 1]
    rogue_rows  = [r for r in rows if int(r["is_rogue"]) == 1]
    if rogue_rows:
        types_seen  = [r["rogue_type"] for r in rogue_rows]
        types_det   = [r["rogue_type"] for r in rogue_rows if int(r["anomaly"]) == 1]
        all_types   = sorted(set(types_seen))
        counts_seen = [types_seen.count(t)  for t in all_types]
        counts_det  = [types_det.count(t)   for t in all_types]
        x_pos       = np.arange(len(all_types))
        ax.bar(x_pos - 0.2, counts_seen, 0.4, label="Seen",     color="#f28e2b", alpha=0.8)
        ax.bar(x_pos + 0.2, counts_det,  0.4, label="Detected", color="#e15759", alpha=0.8)
        ax.set_xticks(x_pos); ax.set_xticklabels(all_types, rotation=10)
        ax.set_ylabel("Advert count"); ax.set_title("Rogue Detection by Type")
        ax.legend(fontsize=8)
    else:
        ax.set_title("Rogue types (no rogue data yet)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out}")
    return out

# ══════════════════════════════════════════════════════════════════════════════
# Input thread (reads commands from stdin while sim runs)
# ══════════════════════════════════════════════════════════════════════════════
class InputThread(threading.Thread):
    def __init__(self, cmd_queue: queue.Queue):
        super().__init__(daemon=True)
        self.cmd_queue = cmd_queue

    def run(self):
        while state.running:
            try:
                raw = input()
            except EOFError:
                break
            parsed = parse_command(raw.strip().lower())
            if parsed is None:
                print(f"  Unknown command '{raw}'. Type 'h' for help.")
                continue
            action = parsed[0]
            if action == "quit":
                state.running = False
                break
            elif action == "help":
                self._print_help()
            elif action == "stats":
                with state.lock:
                    print(json.dumps(state.stats, indent=2))
            else:
                self.cmd_queue.put(parsed)

    def _print_help(self):
        print("""
Commands:
  1+[N]    add N static beacons     (default N=1)
  1-[N]    remove N static beacons
  2+[N]    add N mobile devices
  2-[N]    remove N mobile devices
  3+[s|e|r] inject rogue: s=spoof_uid e=erratic r=replay
  3-[N]    remove N rogues
  s        show current JSON stats
  q        stop simulation and generate plots
  h        this help
""")

# ══════════════════════════════════════════════════════════════════════════════
# Display refresh thread
# ══════════════════════════════════════════════════════════════════════════════
class DisplayThread(threading.Thread):
    def __init__(self, start_real: float):
        super().__init__(daemon=True)
        self.start_real = start_real

    def run(self):
        while state.running:
            render_status(self.start_real)
            time.sleep(2.0)

# ══════════════════════════════════════════════════════════════════════════════
# Main application
# ══════════════════════════════════════════════════════════════════════════════
def main():
    global _data_mgr_ref, _session_id_ref

    session_id       = datetime.now().strftime("%Y%m%d_%H%M%S")
    _session_id_ref  = session_id
    real_start       = time.time()

    print(f"{ANSI_BOLD}BLE Beacon Monitor — Kathmandu (Thamel){ANSI_RESET}")
    print(f"Session ID : {session_id}")
    print(f"Data dir   : {DATA_DIR.resolve()}")

    # Load previous session count for continuity message
    prev_sessions = DataManager.load_sessions()
    prev_adverts  = sum(s.get("total_adverts", 0) for s in prev_sessions)
    if prev_sessions:
        print(f"Previous   : {len(prev_sessions)} session(s), "
              f"{prev_adverts:,} total adverts preserved")

    print("\nBuilding engine…")
    config = {
        "num_static":     8,
        "num_mobile":     4,
        "rogue_percent": 10.0,
        "duration_hours": 999999.0,  # run indefinitely
        "width":          SIM_WIDTH,
        "height":         SIM_HEIGHT,
        "rssi_th":        10.0,
        "int_th":          0.1,
        "sim_th":          0.8,
    }
    engine = _lib.create_engine(json.dumps(config).encode())
    _lib.set_advert_callback(engine, on_advert)
    _lib.set_device_callback(engine, on_device_event)

    data_mgr        = DataManager(session_id)
    _data_mgr_ref   = data_mgr

    cmd_queue = queue.Queue()

    sim_thread     = SimulationThread(engine, data_mgr, cmd_queue, session_id)
    input_thread   = InputThread(cmd_queue)
    display_thread = DisplayThread(real_start)

    # Ctrl+C → graceful shutdown
    def _sigint(sig, frame):
        state.running = False

    signal.signal(signal.SIGINT, _sigint)

    print("Starting simulation. Type 'h' for help or 'q' to stop.\n")
    sim_thread.start()
    display_thread.start()
    input_thread.start()

    # Wait until stopped
    while state.running:
        time.sleep(0.2)

    # ── Shutdown ──────────────────────────────────────────────────────────────
    print(f"\n\n{ANSI_YELLOW}Stopping simulation…{ANSI_RESET}")
    state.running = False

    # Final stats
    with state.lock:
        raw   = _lib.get_stats_json(engine)
        stats = json.loads(raw.decode())

    _lib.destroy_engine(engine)

    real_end      = time.time()
    real_duration = real_end - real_start
    total_adv     = state.total_adverts

    print(f"Simulation stopped.")
    print(f"  Real time elapsed : {fmt_time(real_duration)}")
    print(f"  Sim time elapsed  : {fmt_time(stats.get('time', 0))}")
    print(f"  Total adverts     : {total_adv:,}")
    print(f"  Final device count: {stats.get('device_count', 0)}")

    # Flush all buffered CSV data
    print("Flushing data to CSV…")
    data_mgr.flush_all()

    # Save session metadata
    data_mgr.save_session_meta({
        "session_id":    session_id,
        "start_time":    datetime.fromtimestamp(real_start).isoformat(),
        "end_time":      datetime.fromtimestamp(real_end).isoformat(),
        "duration":      fmt_time(real_duration),
        "total_adverts": total_adv,
        "final_stats":   stats,
    })

    print(f"Data saved → {ADVERTS_CSV}")
    print(f"Session log → {SESSIONS_JS}")

    # Generate plots from all accumulated data
    plot_file = generate_plots(session_id)
    if plot_file:
        print(f"\n{ANSI_GREEN}Done! Open {plot_file} to view results.{ANSI_RESET}")


if __name__ == "__main__":
    main()
