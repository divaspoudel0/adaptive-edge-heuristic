import os, sys

os.chdir("/content/adaptive-edge-heuristic")

# ── Step 1: Recompile to be safe ──────────────────────────────────────────────
os.system("rm -f cpp/libheuristic.so")
ret = os.system(
    "g++ -O3 -shared -fPIC -std=c++11 "
    "-o cpp/libheuristic.so "
    "cpp/engine.cpp cpp/simulator.cpp cpp/learner.cpp "
    "-Icpp"
)
if ret != 0:
    raise RuntimeError("Compilation failed — check errors above")

so = "/content/adaptive-edge-heuristic/cpp/libheuristic.so"
assert os.path.exists(so), "Library file still missing after compile"
print("✅ Library compiled:", so)

# ── Step 2: Overwrite colab_runner.py with absolute path baked in ─────────────
colab_runner_code = r'''
import ctypes, csv, json, os, queue, sys, time, threading
from datetime import datetime
from pathlib import Path

IN_COLAB = False
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    pass

if IN_COLAB:
    import ipywidgets as widgets
    from IPython.display import display, clear_output, Image as IPImage

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

# ── Kathmandu Thamel bounds ───────────────────────────────────────────────────
KTM_LAT_SW = 27.7100; KTM_LON_SW = 85.3070
KTM_LAT_NE = 27.7155; KTM_LON_NE = 85.3128
SIM_WIDTH  = 500.0;   SIM_HEIGHT  = 500.0
LANDMARKS  = {
    "Thamel Chowk":     (0.50, 0.50),
    "Garden of Dreams": (0.82, 0.45),
    "Kathmandu Mall":   (0.30, 0.20),
    "Rani Pokhari":     (0.90, 0.10),
    "Durbar Marg":      (0.70, 0.05),
}

def xy_to_latlon(x, y):
    lat = KTM_LAT_SW + (y/SIM_HEIGHT)*(KTM_LAT_NE-KTM_LAT_SW)
    lon = KTM_LON_SW + (x/SIM_WIDTH) *(KTM_LON_NE-KTM_LON_SW)
    return round(lat,6), round(lon,6)

def nearest_landmark(x, y):
    nx, ny = x/SIM_WIDTH, y/SIM_HEIGHT
    best, bd = "", 1e9
    for n,(lx,ly) in LANDMARKS.items():
        d=((nx-lx)**2+(ny-ly)**2)**0.5
        if d<bd: bd,best=d,n
    return best

# ── Load library using ABSOLUTE path ─────────────────────────────────────────
_LIB_PATH = "/content/adaptive-edge-heuristic/cpp/libheuristic.so"

def _load_lib():
    if not os.path.exists(_LIB_PATH):
        raise RuntimeError(f"Library not found at {_LIB_PATH}\n"
                           "Run the compile cell first.")
    return ctypes.CDLL(_LIB_PATH)

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

for _name, _at, _rt in [
    ("create_engine",       [ctypes.c_char_p],                                   ctypes.c_void_p),
    ("destroy_engine",      [ctypes.c_void_p],                                   None),
    ("set_advert_callback", [ctypes.c_void_p, ADVERT_CB],                        None),
    ("set_device_callback", [ctypes.c_void_p, DEVICE_CB],                        None),
    ("run_step",            [ctypes.c_void_p, ctypes.c_double],                  ctypes.c_int),
    ("add_static_devices",  [ctypes.c_void_p, ctypes.c_int],                     ctypes.c_int),
    ("add_mobile_devices",  [ctypes.c_void_p, ctypes.c_int],                     ctypes.c_int),
    ("remove_devices",      [ctypes.c_void_p, ctypes.c_int, ctypes.c_int],       ctypes.c_int),
    ("inject_rogue_now",    [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_double], ctypes.c_int),
    ("get_device_count",    [ctypes.c_void_p],                                   ctypes.c_int),
    ("get_stats_json",      [ctypes.c_void_p],                                   ctypes.c_char_p),
    ("update_thresholds",   [ctypes.c_void_p, ctypes.c_double,
                              ctypes.c_double, ctypes.c_double],                 None),
]:
    fn = getattr(_lib, _name)
    fn.argtypes = _at
    fn.restype  = _rt

# ── Data persistence ──────────────────────────────────────────────────────────
_BASE      = Path("/content/adaptive-edge-heuristic")
DATA_DIR   = _BASE / "data"
ADVERTS_CSV= DATA_DIR / "adverts.csv"
EVENTS_CSV = DATA_DIR / "events.csv"
SESSIONS_JS= DATA_DIR / "sessions.json"

ADVERT_FIELDS = ["session_id","timestamp","mac","uid","service_id",
                 "rssi","x","y","lat","lon","landmark",
                 "is_rogue","rogue_type","logical_id","anomaly"]
EVENT_FIELDS  = ["session_id","sim_time","real_time",
                 "event","device_id","device_type","total_count"]

def _ensure_header(path, fields):
    if not path.exists():
        with open(path,"w",newline="") as f:
            csv.writer(f).writerow(fields)

_abuf=[]; _ebuf=[]; _bloc=threading.Lock()
_sid=""; _rstart=0.0

def _flush():
    global _abuf, _ebuf
    with _bloc:
        if _abuf:
            with open(ADVERTS_CSV,"a",newline="") as f:
                csv.DictWriter(f,ADVERT_FIELDS,extrasaction="ignore").writerows(_abuf)
            _abuf=[]
        if _ebuf:
            with open(EVENTS_CSV,"a",newline="") as f:
                csv.DictWriter(f,EVENT_FIELDS,extrasaction="ignore").writerows(_ebuf)
            _ebuf=[]

# ── Shared state ──────────────────────────────────────────────────────────────
class _S:
    lock=threading.Lock(); stats={}; rogue_alerts=[]
    total_adverts=0; running=False; engine=None
_st=_S()

# ── Callbacks ─────────────────────────────────────────────────────────────────
@ADVERT_CB
def _on_advert(ts,mac,uid,svc,rssi,x,y,is_rogue,rtype,lid,anomaly):
    lat,lon=xy_to_latlon(x,y); lm=nearest_landmark(x,y)
    rt=rtype.decode() if rtype else ""
    with _st.lock: _st.total_adverts+=1
    if is_rogue and anomaly:
        msg=(f"[{datetime.now().strftime('%H:%M:%S')}] "
             f"{rt} near {lm} ({lat}N,{lon}E) RSSI={rssi:.1f}dBm")
        with _st.lock:
            _st.rogue_alerts.append(msg)
            if len(_st.rogue_alerts)>10: _st.rogue_alerts.pop(0)
    with _bloc:
        _abuf.append({"session_id":_sid,"timestamp":round(ts,3),
            "mac":mac.decode(),"uid":uid.decode(),"service_id":svc.decode(),
            "rssi":round(rssi,2),"x":round(x,2),"y":round(y,2),
            "lat":lat,"lon":lon,"landmark":lm,
            "is_rogue":is_rogue,"rogue_type":rt,"logical_id":lid,"anomaly":anomaly})
        if len(_abuf)>=500: _flush()

@DEVICE_CB
def _on_device(event,device_id,device_type,total):
    with _bloc:
        _ebuf.append({"session_id":_sid,"sim_time":_st.stats.get("time",0),
            "real_time":datetime.now().isoformat(),"event":event.decode(),
            "device_id":device_id.decode(),"device_type":device_type.decode(),
            "total_count":total})

# ── Sim thread ────────────────────────────────────────────────────────────────
_cq=queue.Queue()

def _sim_loop():
    ctr=0
    while _st.running:
        while not _cq.empty():
            try:
                act,dt,cnt,ex=_cq.get_nowait()
                with _st.lock: _apply(act,dt,cnt,ex)
            except queue.Empty: break
        with _st.lock: _lib.run_step(_st.engine, 0.1)
        ctr+=1
        if ctr%200==0:
            raw=_lib.get_stats_json(_st.engine)
            with _st.lock: _st.stats=json.loads(raw.decode())
        time.sleep(0.0005)

def _apply(action,dev_type,count,extra):
    e=_st.engine
    if action=="add":
        if   dev_type==1: _lib.add_static_devices(e,count)
        elif dev_type==2: _lib.add_mobile_devices(e,count)
        elif dev_type==3:
            rt=(extra or "spoof_uid").encode()
            for _ in range(count):
                _lib.inject_rogue_now(e,rt,ctypes.c_double(300.0))
    elif action=="remove":
        _lib.remove_devices(e,count,1 if dev_type==3 else 0)

# ── Plot ──────────────────────────────────────────────────────────────────────
def _plot(session_id):
    if not ADVERTS_CSV.exists(): print("No data yet."); return None
    with open(ADVERTS_CSV,newline="") as f: rows=list(csv.DictReader(f))
    with open(EVENTS_CSV, newline="") as f: evts=list(csv.DictReader(f))
    if not rows: print("Empty CSV."); return None

    def col(r,k,t=float): return np.array([t(x[k]) for x in r if k in x],dtype=t)
    ts=col(rows,"timestamp"); rssi=col(rows,"rssi")
    anomaly=col(rows,"anomaly",int); truth=col(rows,"is_rogue",int)
    lids=col(rows,"logical_id",int)
    lat_a=col(rows,"lat"); lon_a=col(rows,"lon")
    ev_t=[float(e["sim_time"]) for e in evts]
    ev_ty=[e["device_type"] for e in evts]
    ev_ev=[e["event"] for e in evts]

    fig,axes=plt.subplots(4,2,figsize=(18,20))
    fig.suptitle(f"BLE Monitor — Kathmandu (Thamel)\nTotal adverts: {len(rows):,}",
                 fontsize=14,fontweight="bold")

    ax=axes[0,0]
    nm=truth==0; rg=truth==1
    if nm.any(): ax.plot(ts[nm],rssi[nm],"b.",ms=1,alpha=0.2,label="Normal")
    if rg.any(): ax.plot(ts[rg],rssi[rg],"rx",ms=2,alpha=0.6,label="Rogue")
    ax.set_xlabel("Sim time (s)"); ax.set_ylabel("RSSI (dBm)")
    ax.set_title("RSSI over Time"); ax.legend(markerscale=4,fontsize=8)

    ax=axes[0,1]
    if nm.any(): ax.scatter(lon_a[nm],lat_a[nm],s=2,c="steelblue",alpha=0.15,label="Normal")
    if rg.any(): ax.scatter(lon_a[rg],lat_a[rg],s=6,c="red",alpha=0.5,label="Rogue")
    det=rg&(anomaly==1)
    if det.any(): ax.scatter(lon_a[det],lat_a[det],s=20,c="orange",marker="*",label="Detected")
    for name,(rx,ry) in LANDMARKS.items():
        plat=KTM_LAT_SW+ry*(KTM_LAT_NE-KTM_LAT_SW)
        plon=KTM_LON_SW+rx*(KTM_LON_NE-KTM_LON_SW)
        ax.annotate(name,(plon,plat),fontsize=6,color="gray",ha="center")
        ax.plot(plon,plat,"k+",ms=5)
    ax.set_xlim(KTM_LON_SW,KTM_LON_NE); ax.set_ylim(KTM_LAT_SW,KTM_LAT_NE)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Device Map — Thamel, Kathmandu")
    ax.legend(markerscale=4,fontsize=7)

    ax=axes[1,0]
    win=max(500,len(anomaly)//50)
    if len(anomaly)>win:
        rate=np.convolve(anomaly.astype(float),np.ones(win)/win,mode="valid")
        ax.plot(ts[win-1:],rate*100,color="#e15759",lw=1.2)
    ax.set_xlabel("Sim time (s)"); ax.set_ylabel("Anomaly rate (%)")
    ax.set_title(f"Rolling Anomaly Rate (window={win})")

    ax=axes[1,1]
    tc={"static":"#4e79a7","mobile":"#f28e2b","rogue":"#e15759"}
    if ev_t:
        for t,dt,ev in zip(ev_t,ev_ty,ev_ev):
            ax.scatter(t,1 if ev=="added" else -1,color=tc.get(dt,"gray"),s=8,alpha=0.5)
        ax.axhline(0,color="gray",lw=0.5)
        ax.set_yticks([1,-1]); ax.set_yticklabels(["Added","Removed"])
        ax.set_xlabel("Sim time (s)"); ax.set_title("Device Event Timeline")
        ax.legend(handles=[mpatches.Patch(color=c,label=t) for t,c in tc.items()],fontsize=8)

    ax=axes[2,0]
    if HAS_SKLEARN and len(np.unique(truth))>1:
        cm=confusion_matrix(truth,anomaly)
        disp=ConfusionMatrixDisplay(cm,display_labels=["Normal","Rogue"])
        disp.plot(ax=ax,cmap="Blues",colorbar=False)
        tn,fp,fn,tp=cm.ravel()
        prec=tp/(tp+fp) if tp+fp>0 else 0
        rec=tp/(tp+fn) if tp+fn>0 else 0
        f1=2*prec*rec/(prec+rec) if prec+rec>0 else 0
        ax.set_title(f"Confusion Matrix P={prec:.2f} R={rec:.2f} F1={f1:.2f}")
    else:
        ax.set_title("Confusion Matrix (need both classes)")

    ax=axes[2,1]
    if len(lids)>0:
        ul,uc=np.unique(lids[lids>=0],return_counts=True)
        top=min(40,len(ul)); idx=np.argsort(uc)[-top:][::-1]
        ax.bar(range(top),uc[idx],color="#4e79a7",alpha=0.8)
        ax.set_xlabel("Logical device rank"); ax.set_ylabel("Advert count")
        ax.set_title(f"Top {top} Logical Devices")

    ax=axes[3,0]; ax.axis("off")
    if SESSIONS_JS.exists():
        try:
            metas=json.loads(SESSIONS_JS.read_text())
            headers=["Session","Start","Adverts","Duration"]
            cells=[[m.get("session_id","")[:12],m.get("start_time","")[:19],
                    f"{m.get('total_adverts',0):,}",m.get("duration","")]
                   for m in metas[-8:]]
            tbl=ax.table(cellText=cells,colLabels=headers,loc="center",cellLoc="center")
            tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1,1.4)
            ax.set_title("Session History",pad=20)
        except Exception: pass

    ax=axes[3,1]
    rr=[r for r in rows if int(r["is_rogue"])==1]
    if rr:
        ts2=[r["rogue_type"] for r in rr]
        td=[r["rogue_type"] for r in rr if int(r["anomaly"])==1]
        at=sorted(set(ts2)); xp=np.arange(len(at))
        ax.bar(xp-0.2,[ts2.count(t) for t in at],0.4,label="Seen",color="#f28e2b",alpha=0.8)
        ax.bar(xp+0.2,[td.count(t)  for t in at],0.4,label="Detected",color="#e15759",alpha=0.8)
        ax.set_xticks(xp); ax.set_xticklabels(at,rotation=10)
        ax.set_ylabel("Advert count"); ax.set_title("Rogue Detection by Type")
        ax.legend(fontsize=8)
    else:
        ax.set_title("Rogue types (no rogue data yet)")

    plt.tight_layout(rect=[0,0,1,0.96])
    out=str(_BASE/f"simulation_results_{session_id}.png")
    plt.savefig(out,dpi=150,bbox_inches="tight"); plt.close()
    print(f"Plot saved → {out}")
    return out

# ── Colab UI ──────────────────────────────────────────────────────────────────
def _build_ui():
    dd_type=widgets.Dropdown(
        options=[("1 — Static beacon",1),("2 — Mobile device",2),("3 — Rogue device",3)],
        description="Type:",layout=widgets.Layout(width="230px"))
    dd_cnt=widgets.Dropdown(options=[1,2,5,10,20],description="Count:",
        layout=widgets.Layout(width="150px"))
    dd_rogue=widgets.Dropdown(
        options=[("spoof_uid","spoof_uid"),("erratic_timing","erratic_timing"),("replay","replay")],
        description="Rogue type:",layout=widgets.Layout(width="250px"))
    btn_add=widgets.Button(description="+ Add",   button_style="success",layout=widgets.Layout(width="100px"))
    btn_rem=widgets.Button(description="- Remove",button_style="warning",layout=widgets.Layout(width="100px"))
    btn_stp=widgets.Button(description="Stop & Plot",button_style="danger",layout=widgets.Layout(width="130px"))
    btn_st =widgets.Button(description="Stats",   button_style="info",   layout=widgets.Layout(width="100px"))
    log_out  =widgets.Output(layout=widgets.Layout(height="180px",overflow_y="scroll",border="1px solid #ccc",padding="6px"))
    stats_out=widgets.Output(layout=widgets.Layout(height="110px",overflow_y="scroll",border="1px solid #ddd",padding="6px"))
    status   =widgets.HTML(value="<b>Starting...</b>")

    def _log(msg):
        with log_out: print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    def _on_add(_):
        dt=dd_type.value; n=dd_cnt.value; ex=dd_rogue.value if dt==3 else None
        _cq.put(("add",dt,n,ex)); _log(f"Add {n} × type={dt}" + (f" [{ex}]" if ex else ""))
    def _on_rem(_):
        _cq.put(("remove",dd_type.value,dd_cnt.value,None))
        _log(f"Remove {dd_cnt.value} × type={dd_type.value}")
    def _on_stp(_): _st.running=False; _log("Stopping...")
    def _on_st(_):
        with _st.lock: s=dict(_st.stats)
        with stats_out: clear_output(wait=True); print(json.dumps(s,indent=2))

    btn_add.on_click(_on_add); btn_rem.on_click(_on_rem)
    btn_stp.on_click(_on_stp); btn_st.on_click(_on_st)

    def _refresh():
        while _st.running:
            with _st.lock:
                s=dict(_st.stats); ta=_st.total_adverts; al=list(_st.rogue_alerts)
            elapsed=time.time()-_rstart
            h=int(elapsed)//3600; m=(int(elapsed)%3600)//60; sec=int(elapsed)%60
            rc=s.get("rogue_count",0)
            rc_col="red" if rc>0 else "green"
            rc_txt="⚠ ROGUES ACTIVE" if rc>0 else "✓ CLEAN"
            html=(f"<div style='font-family:monospace;font-size:13px;line-height:1.8'>"
                  f"<b>Elapsed:</b> {h:02d}h {m:02d}m {sec:02d}s &nbsp;|&nbsp; "
                  f"<b>Sim time:</b> {int(s.get('time',0))//3600:02d}h "
                  f"{(int(s.get('time',0))%3600)//60:02d}m<br>"
                  f"<b>Devices:</b> Static={s.get('static_count',0)} "
                  f"Mobile={s.get('mobile_count',0)} "
                  f"Rogues=<span style='color:{rc_col}'><b>{rc}</b></span> "
                  f"Total={s.get('device_count',0)} "
                  f"<span style='color:{rc_col}'><b>{rc_txt}</b></span><br>"
                  f"<b>Adverts:</b> {ta:,} &nbsp;|&nbsp; "
                  f"<b>Anomaly:</b> {s.get('anomaly_rate',0)*100:.1f}% "
                  f"FP={s.get('fp_rate',0)*100:.1f}% "
                  f"FN={s.get('fn_rate',0)*100:.1f}%<br>"
                  f"<b>Thresholds:</b> RSSI={s.get('rssi_th',0):.2f} "
                  f"Interval={s.get('int_th',0):.3f}</div>")
            if al:
                html+="<hr>"+"<br>".join(f"<span style='color:red'>⚠ {a}</span>" for a in al[-3:])
            status.value=html
            time.sleep(2.0)

    threading.Thread(target=_refresh,daemon=True).start()
    ui=widgets.VBox([
        widgets.HTML("<h3>🛰 BLE Beacon Monitor — Kathmandu (Thamel)</h3>"),
        status,
        widgets.HBox([dd_type,dd_cnt,dd_rogue]),
        widgets.HBox([btn_add,btn_rem,btn_st,btn_stp]),
        widgets.HTML("<b>Activity log:</b>"),log_out,
        widgets.HTML("<b>Stats:</b>"),stats_out,
    ])
    display(ui)
    return _log

# ── Public entry point ────────────────────────────────────────────────────────
def start(num_static=8, num_mobile=4, rogue_percent=10.0):
    global _sid, _rstart
    DATA_DIR.mkdir(exist_ok=True)
    _ensure_header(ADVERTS_CSV,ADVERT_FIELDS)
    _ensure_header(EVENTS_CSV,EVENT_FIELDS)
    _sid=datetime.now().strftime("%Y%m%d_%H%M%S")
    _rstart=time.time()

    config={"num_static":num_static,"num_mobile":num_mobile,
            "rogue_percent":rogue_percent,"duration_hours":999999.0,
            "width":SIM_WIDTH,"height":SIM_HEIGHT,
            "rssi_th":10.0,"int_th":0.1,"sim_th":0.8}
    engine=_lib.create_engine(json.dumps(config).encode())
    _lib.set_advert_callback(engine,_on_advert)
    _lib.set_device_callback(engine,_on_device)
    with _st.lock: _st.engine=engine; _st.running=True

    threading.Thread(target=_sim_loop,daemon=True).start()

    if IN_COLAB:
        _log=_build_ui()
        _log(f"Session {_sid} started: {num_static} static + {num_mobile} mobile")
        while _st.running: time.sleep(0.5)
    else:
        print(f"Session {_sid} running. Ctrl+C to stop.")
        try:
            while True: time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        _st.running=False

    with _st.lock:
        raw=_lib.get_stats_json(engine); stats=json.loads(raw.decode())
    _lib.destroy_engine(engine)
    _flush()

    dur=time.time()-_rstart
    h=int(dur)//3600; m=(int(dur)%3600)//60; s=int(dur)%60
    meta={"session_id":_sid,
          "start_time":datetime.fromtimestamp(_rstart).isoformat(),
          "end_time":datetime.now().isoformat(),
          "duration":f"{h:02d}h {m:02d}m {s:02d}s",
          "total_adverts":_st.total_adverts,"final_stats":stats}
    existing=[]
    if SESSIONS_JS.exists():
        try: existing=json.loads(SESSIONS_JS.read_text())
        except Exception: pass
    existing.append(meta)
    SESSIONS_JS.write_text(json.dumps(existing,indent=2))
    print(f"Session ended. Adverts: {_st.total_adverts:,}")

    out=_plot(_sid)
    if out and IN_COLAB:
        display(IPImage(out))
    return out
