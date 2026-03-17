#pragma once
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* EngineHandle;

// ─── Advertisement callback ───────────────────────────────────────────────────
// Fields: timestamp, mac, uid, service_id, rssi, x, y,
//         is_rogue(0/1), rogue_type, logical_id, anomaly(0/1)
typedef void (*AdvertCallback)(double, const char*, const char*, const char*,
                               double, double, double, int, const char*, int, int);

// ─── Device-event callback ────────────────────────────────────────────────────
// Called whenever a device is added or removed.
// Fields: event("added"/"removed"), device_id, device_type("static"/"mobile"/"rogue"),
//         current_total_count
typedef void (*DeviceEventCallback)(const char*, const char*, const char*, int);

// ─── Lifecycle ────────────────────────────────────────────────────────────────
EngineHandle  create_engine          (const char* config_json);
void          destroy_engine         (EngineHandle engine);

// ─── Callbacks ────────────────────────────────────────────────────────────────
void          set_advert_callback    (EngineHandle engine, AdvertCallback      cb);
void          set_device_callback    (EngineHandle engine, DeviceEventCallback cb);

// ─── Simulation step ──────────────────────────────────────────────────────────
// Returns 1 while simulation is running, 0 when time is exhausted.
int           run_step               (EngineHandle engine, double dt);

// ─── Dynamic device management (safe to call between run_step() calls) ────────
// Add devices at runtime. Returns new total device count.
int           add_static_devices     (EngineHandle engine, int count);
int           add_mobile_devices     (EngineHandle engine, int count);

// Remove devices at runtime.
// rogue_only=1 → remove only rogue devices; rogue_only=0 → remove any non-static.
// Returns new total device count.
int           remove_devices         (EngineHandle engine, int count, int rogue_only);

// Remove a specific device by its string ID. Returns 1 if found and removed.
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

// Returns JSON: time, device counts, anomaly_rate, thresholds, logical_count, etc.
const char*   get_stats_json         (EngineHandle engine);

// ─── Threshold control ────────────────────────────────────────────────────────
void          update_thresholds      (EngineHandle engine,
                                      double rssi_th, double int_th, double sim_th);

#ifdef __cplusplus
}
#endif