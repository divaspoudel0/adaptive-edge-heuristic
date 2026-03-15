#pragma once
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* EngineHandle;

// Callback: timestamp, mac, uid, service_id, rssi, x, y,
//           is_rogue(0/1), rogue_type, logical_id, anomaly(0/1)
typedef void (*AdvertCallback)(double, const char*, const char*, const char*,
                               double, double, double, int, const char*, int, int);

EngineHandle  create_engine      (const char* config_json);
void          destroy_engine     (EngineHandle engine);
void          set_callback       (EngineHandle engine, AdvertCallback callback);
int           run_step           (EngineHandle engine, double dt);   // returns 1 while running
const char*   get_stats_json     (EngineHandle engine);
void          update_thresholds  (EngineHandle engine, double rssi_th, double int_th, double sim_th);

#ifdef __cplusplus
}
#endif
