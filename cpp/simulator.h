#pragma once
#include <string>
#include <vector>
#include <functional>
#include <random>
#include <tuple>

struct Advert {
    double      timestamp;
    std::string mac;
    std::string uid;
    std::string service_id;
    double      rssi;
    double      x, y;
    bool        is_rogue;
    std::string rogue_type;
    int         logical_id;
    int         anomaly;
};

struct DeviceEvent {
    std::string event;        // "added" | "removed"
    std::string device_id;
    std::string device_type;  // "static" | "mobile" | "rogue"
    int         total_count;
};

class Device {
public:
    // Declaration order matches constructor initializer list (avoids -Wreorder)
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
    double      end_time;
    std::string mac;
    std::string uid;
    bool        is_mobile;

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

class BeaconSimulator {
public:
    BeaconSimulator(int    num_static,
                    int    num_mobile,
                    double rogue_percent,
                    double duration_hours,   // use 999999 for indefinite
                    double width,
                    double height);

    void step(double dt,
              std::function<void(const Advert&)>      advert_cb,
              std::function<void(const DeviceEvent&)> device_cb);

    // ── Dynamic management (call between steps) ───────────────────────────────
    int addStaticDevices (int count);
    int addMobileDevices (int count);
    int removeDevices    (int count, bool rogue_only);
    int removeDeviceById (const std::string& id);

    // Public so engine can call on behalf of inject_rogue_now
    void injectRogue(double current_time,
                     double duration,
                     const std::string& type);

    // ── Queries ───────────────────────────────────────────────────────────────
    int    deviceCount() const;
    int    staticCount() const;
    int    mobileCount() const;
    int    rogueCount () const;
    double currentTime() const { return current_time_; }
    bool   isRunning  () const { return current_time_ < total_duration_; }

private:
    double current_time_;
    double total_duration_;
    double width_, height_;
    int    device_serial_;

    std::vector<Device>    devices_;
    std::mt19937           rng_;

    std::vector<std::tuple<double, double, std::string>> rogue_schedule_;
    std::vector<DeviceEvent> pending_events_;

    Device makeStaticDevice(const std::string& id);
    Device makeMobileDevice(const std::string& id);
    void   removeExpiredRogues(double current_time);
    void   emitEvent(const std::string& event,
                     const std::string& device_id,
                     const std::string& device_type);
};
