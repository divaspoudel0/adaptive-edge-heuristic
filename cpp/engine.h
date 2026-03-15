#pragma once
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* EngineHandle;

// ─── Advertisement callback ───────────────────────────────────────────────────
// timestamp, mac, uid, service_id, rssi, x, y,
// is_rogue(0/1), rogue_type, logical_id, anomaly(0/1)
typedef void (*AdvertCallback)(double, const char*, const char*, const char*,
                               double, double, double, int, const char*, int, int);

// ─── Device-event callback ────────────────────────────────────────────────────
// event("added"/"removed"), device_id, device_type("static"/"mobile"/"rogue"),
// current_total_count
typedef void (*DeviceEventCallback)(const char*, const char*, const char*, int);

// ─── Lifecycle ────────────────────────────────────────────────────────────────
EngineHandle  create_engine          (const char* config_json);
void          destroy_engine         (EngineHandle engine);

// ─── Callbacks ────────────────────────────────────────────────────────────────
void          set_advert_callback    (EngineHandle engine, AdvertCallback      cb);
void          set_device_callback    (EngineHandle engine, DeviceEventCallback cb);

// ─── Simulation step ──────────────────────────────────────────────────────────
// Returns 1 always (runs indefinitely until destroy_engine).
int           run_step               (EngineHandle engine, double dt);

// ─── Dynamic device management ───────────────────────────────────────────────
int           add_static_devices     (EngineHandle engine, int count);
int           add_mobile_devices     (EngineHandle engine, int count);
// rogue_only=1 → target rogues first; 0 → target mobile devices
int           remove_devices         (EngineHandle engine, int count, int rogue_only);
int           remove_device_by_id    (EngineHandle engine, const char* device_id);

// Immediately inject a rogue device.
// rogue_type: "spoof_uid" | "erratic_timing" | "replay"
// duration_sec: lifetime of the rogue (seconds of sim time)
// Returns new total device count.
int           inject_rogue_now       (EngineHandle engine,
                                      const char* rogue_type,
                                      double duration_sec);

// ─── Queries ──────────────────────────────────────────────────────────────────
int           get_device_count       (EngineHandle engine);
int           get_static_count       (EngineHandle engine);
int           get_mobile_count       (EngineHandle engine);
int           get_rogue_count        (EngineHandle engine);

// JSON: time, device_count, static_count, mobile_count, rogue_count,
//       logical_count, anomaly_rate, fp_rate, fn_rate,
//       rssi_th, int_th, sim_th
const char*   get_stats_json         (EngineHandle engine);

// ─── Threshold control ────────────────────────────────────────────────────────
void          update_thresholds      (EngineHandle engine,
                                      double rssi_th, double int_th, double sim_th);

#ifdef __cplusplus
}
#endif
