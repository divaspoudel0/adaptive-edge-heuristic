#pragma once
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* EngineHandle;

typedef void (*AdvertCallback)(double, const char*, const char*, const char*,
                               double, double, double, int, const char*, int, int);

typedef void (*DeviceEventCallback)(const char*, const char*, const char*, int);

EngineHandle  create_engine          (const char* config_json);
void          destroy_engine         (EngineHandle engine);
void          set_advert_callback    (EngineHandle engine, AdvertCallback      cb);
void          set_device_callback    (EngineHandle engine, DeviceEventCallback cb);
int           run_step               (EngineHandle engine, double dt);
int           add_static_devices     (EngineHandle engine, int count);
int           add_mobile_devices     (EngineHandle engine, int count);
int           remove_devices         (EngineHandle engine, int count, int rogue_only);
int           remove_device_by_id    (EngineHandle engine, const char* device_id);
int           inject_rogue_now       (EngineHandle engine,
                                      const char* rogue_type,
                                      double duration_sec);
int           get_device_count       (EngineHandle engine);
int           get_static_count       (EngineHandle engine);
int           get_mobile_count       (EngineHandle engine);
int           get_rogue_count        (EngineHandle engine);
const char*   get_stats_json         (EngineHandle engine);
void          update_thresholds      (EngineHandle engine,
                                      double rssi_th, double int_th, double sim_th);

#ifdef __cplusplus
}
#endif
