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

// ─── Device ──────────────────────────────────────────────────────────────────
class Device {
public:
    // Declaration order matches constructor initializer list to avoid -Wreorder
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
    double      end_time;     // FIX: was missing; set to 1e18 for legit devices
    std::string mac;
    std::string uid;

    Device(const std::string& id,
           const std::string& svc,
           bool rogue,
           const std::string& rtype,
           double x0, double y0,
           double mac_int, double uid_int, double adv_int);

    // FIX: rng must be passed – Device has no rng member
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

    // FIX: renamed run() -> step() for clarity; engine.cpp was calling step()
    void   step       (double dt, std::function<void(const Advert&)> callback);
    double currentTime() const { return current_time_; }
    bool   isRunning  () const { return current_time_ < total_duration_; }

private:
    double current_time_;
    double total_duration_;
    double width_, height_;

    std::vector<Device>    devices_;
    std::mt19937           rng_;

    // FIX: store (start_time, duration, type) so injectRogue uses scheduled values
    std::vector<std::tuple<double, double, std::string>> rogue_schedule_;

    void injectRogue        (double current_time, double duration, const std::string& type);
    void removeExpiredRogues(double current_time);
};
