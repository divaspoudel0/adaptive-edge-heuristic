#pragma once
#include <string>
#include <vector>
#include <functional>
#include <random>
#include <tuple>

// ─── Advert ──────────────────────────────────────────────────────────────────
struct Advert {
    double      timestamp;
    std::string mac;
    std::string uid;
    std::string service_id;
    double      rssi;
    double      x, y;
    bool        is_rogue;
    std::string rogue_type;
    int         logical_id;   // filled by engine
    int         anomaly;      // filled by engine
};

// ─── Device event (emitted when a device is added or removed) ─────────────────
struct DeviceEvent {
    std::string event;        // "added" | "removed"
    std::string device_id;
    std::string device_type;  // "static" | "mobile" | "rogue"
    int         total_count;
};

// ─── Device ──────────────────────────────────────────────────────────────────
class Device {
public:
    // Declaration order must match constructor initializer list (avoids -Wreorder)
    std::string device_id;
    std::string service_id;
    bool        is_rogue;
    std::string rogue_type;
    double      x, y;
    double      vx, vy;
    double      mac_rotation_interval;
    double      uid_rotation_interval;
    double      advertisement_interval;
    double      last_advertisement;
    double      next_mac_rotation;
    double      next_uid_rotation;
    double      end_time;     // 1e18 for permanent devices; finite for rogues
    std::string mac;
    std::string uid;
    bool        is_mobile;    // true if device has velocity

    Device(const std::string& id,
           const std::string& svc,
           bool rogue,
           const std::string& rtype,
           double x0, double y0,
           double mac_int, double uid_int,
           double adv_int,
           bool mobile);

    void   updatePosition(double dt, double width, double height, std::mt19937& rng);
    void   rotateMac     (double current_time, std::mt19937& rng);
    void   rotateUid     (double current_time, std::mt19937& rng);
    Advert generateAdvert(double current_time, double rx, double ry) const;
};

// ─── BeaconSimulator ─────────────────────────────────────────────────────────
class BeaconSimulator {
public:
    BeaconSimulator(int    num_static,
                    int    num_mobile,
                    double rogue_percent,
                    double duration_hours,
                    double width,
                    double height);

    // Advance simulation by dt seconds; fires callback for every advert emitted.
    // Also fires device_cb for any add/remove events that occur during this step.
    void step(double dt,
              std::function<void(const Advert&)>      advert_cb,
              std::function<void(const DeviceEvent&)> device_cb);

    // ── Dynamic device management ─────────────────────────────────────────────
    // All of these are safe to call between step() calls.
    int addStaticDevices (int count);
    int addMobileDevices (int count);
    int removeDevices    (int count, bool rogue_only);
    int removeDeviceById (const std::string& id);

    // ── Queries ───────────────────────────────────────────────────────────────
    int    deviceCount ()  const;
    int    staticCount ()  const;
    int    mobileCount ()  const;
    int    rogueCount  ()  const;
    double currentTime ()  const { return current_time_; }
    bool   isRunning   ()  const { return current_time_ < total_duration_; }

private:
    double current_time_;
    double total_duration_;
    double width_, height_;
    int    device_serial_;    // monotonic counter for unique IDs

    std::vector<Device> devices_;
    std::mt19937        rng_;

    // Scheduled rogue injections: (start_time, duration, type)
    std::vector<std::tuple<double, double, std::string>> rogue_schedule_;

    // Pending device events to fire on next step()
    std::vector<DeviceEvent> pending_events_;

    // Helpers
    Device makeStaticDevice(const std::string& id);
    Device makeMobileDevice(const std::string& id);
    void   injectRogue     (double current_time,
                            double duration,
                            const std::string& type);
    void   removeExpiredRogues(double current_time);
    void   emitEvent       (const std::string& event,
                            const std::string& device_id,
                            const std::string& device_type);
};
