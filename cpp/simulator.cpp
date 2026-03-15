#include "simulator.h"
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>

// FIX: M_PI is not guaranteed in C++11 strict mode
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// FIX: std::clamp is C++17. Provide a C++11 compatible version.
template<typename T>
static T clamp11(T val, T lo, T hi) {
    return val < lo ? lo : (val > hi ? hi : val);
}

// ─── RSSI model ───────────────────────────────────────────────────────────────
static double rssi_from_distance(double d, double P0 = -59.0, double n = 2.0) {
    if (d < 0.01) d = 0.01;
    return P0 - 10.0 * n * std::log10(d);
}

// ─── Random string helpers ────────────────────────────────────────────────────
static std::string random_mac(std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(0, 255);
    std::ostringstream ss;
    // Locally-administered unicast MAC (02:xx:xx:xx:xx:xx)
    ss << "02";
    for (int i = 0; i < 5; ++i)
        ss << ":" << std::hex << std::setfill('0') << std::setw(2) << dist(rng);
    return ss.str();
}

static std::string random_uid(std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(0, 255);
    std::ostringstream ss;
    // Pseudo-UUID format: 8-4-4-4-12 hex chars
    for (int group = 0; group < 5; ++group) {
        if (group > 0) ss << "-";
        int len = (group == 0) ? 4 : (group < 4) ? 2 : 6;
        for (int i = 0; i < len; ++i)
            ss << std::hex << std::setfill('0') << std::setw(2) << dist(rng);
    }
    return ss.str();
}

// ─── Device implementation ────────────────────────────────────────────────────
Device::Device(const std::string& id,
               const std::string& svc,
               bool rogue,
               const std::string& rtype,
               double x0, double y0,
               double mac_int, double uid_int, double adv_int)
    : device_id(id), service_id(svc), is_rogue(rogue), rogue_type(rtype),
      x(x0), y(y0), vx(0.0), vy(0.0),
      mac_rotation_interval(mac_int),
      uid_rotation_interval(uid_int),
      advertisement_interval(adv_int),
      last_advertisement(-adv_int),    // ensures first advert fires immediately
      next_mac_rotation(mac_int),
      next_uid_rotation(uid_int),
      end_time(1e18)                   // FIX: initialize for all devices (rogues override)
{}

// FIX: rng passed by reference – it is the simulator's rng, not a Device member
void Device::updatePosition(double dt, double width, double height, std::mt19937& rng) {
    // Occasional random direction change (~1% chance per second)
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    if (prob(rng) < 0.01 * dt) {
        std::uniform_real_distribution<double> angle_dist(0.0, 2.0 * M_PI);
        std::uniform_real_distribution<double> speed_dist(0.5, 2.0);
        double angle = angle_dist(rng);
        double speed = speed_dist(rng);
        vx = speed * std::cos(angle);
        vy = speed * std::sin(angle);
    }

    x += vx * dt;
    y += vy * dt;

    // Bounce off walls
    if (x < 0.0)    { vx =  std::abs(vx); x = 0.0;   }
    if (x > width)  { vx = -std::abs(vx); x = width;  }
    if (y < 0.0)    { vy =  std::abs(vy); y = 0.0;    }
    if (y > height) { vy = -std::abs(vy); y = height; }
}

void Device::rotateMac(double current_time, std::mt19937& rng) {
    if (current_time >= next_mac_rotation) {
        mac = random_mac(rng);
        next_mac_rotation = current_time + mac_rotation_interval;
    }
}

void Device::rotateUid(double current_time, std::mt19937& rng) {
    if (current_time >= next_uid_rotation) {
        uid = random_uid(rng);
        next_uid_rotation = current_time + uid_rotation_interval;
    }
}

Advert Device::generateAdvert(double current_time, double rx, double ry) const {
    Advert adv;
    adv.timestamp  = current_time;
    adv.mac        = mac;
    adv.uid        = uid;
    adv.service_id = service_id;
    adv.x          = x;
    adv.y          = y;
    adv.is_rogue   = is_rogue;
    adv.rogue_type = is_rogue ? rogue_type : "";
    adv.logical_id = -1;   // filled by engine
    adv.anomaly    = 0;    // filled by engine

    double dx   = x - rx;
    double dy   = y - ry;
    double dist = std::sqrt(dx * dx + dy * dy);
    adv.rssi    = rssi_from_distance(dist);
    return adv;
}

// ─── BeaconSimulator implementation ──────────────────────────────────────────
BeaconSimulator::BeaconSimulator(int    num_static,
                                 int    num_mobile,
                                 double rogue_percent,
                                 double duration_hours,
                                 double width,
                                 double height)
    : current_time_(0.0),
      total_duration_(duration_hours * 3600.0),
      width_(width), height_(height),
      rng_(std::random_device{}())
{
    // FIX: separate distributions for x and y so height != width works correctly
    std::uniform_real_distribution<double> pos_x(0.0, width_);
    std::uniform_real_distribution<double> pos_y(0.0, height_);

    // ── Static beacons (no velocity, no ID rotation) ─────────────────────────
    for (int i = 0; i < num_static; ++i) {
        std::string svc = "SVC_" + std::to_string(i % 3 + 1);
        Device dev("static_" + std::to_string(i), svc, false, "",
                   pos_x(rng_), pos_y(rng_),
                   1e18, 1e18,   // never rotate
                   0.5);
        dev.mac = random_mac(rng_);
        dev.uid = random_uid(rng_);
        devices_.push_back(dev);
    }

    // ── Mobile devices ────────────────────────────────────────────────────────
    std::uniform_real_distribution<double> speed_dist(0.5, 2.0);
    std::uniform_real_distribution<double> angle_dist(0.0, 2.0 * M_PI);
    std::uniform_real_distribution<double> mac_int_dist(120.0, 300.0);
    std::uniform_real_distribution<double> uid_int_dist(60.0, 180.0);

    for (int i = 0; i < num_mobile; ++i) {
        double angle = angle_dist(rng_);
        double speed = speed_dist(rng_);
        std::string svc = "SVC_" + std::to_string(i % 3 + 1);
        Device dev("mobile_" + std::to_string(i), svc, false, "",
                   pos_x(rng_), pos_y(rng_),
                   mac_int_dist(rng_), uid_int_dist(rng_),
                   0.5);
        dev.vx  = speed * std::cos(angle);
        dev.vy  = speed * std::sin(angle);
        dev.mac = random_mac(rng_);
        dev.uid = random_uid(rng_);
        devices_.push_back(dev);
    }

    // ── Schedule rogue injection events ──────────────────────────────────────
    int total_devices   = num_static + num_mobile;
    int num_rogue_events = std::max(1, static_cast<int>(total_devices * rogue_percent / 100.0));

    // Don't inject in first/last 5% of simulation time
    double t_lo = total_duration_ * 0.05;
    double t_hi = total_duration_ * 0.95;
    std::uniform_real_distribution<double> time_dist(t_lo, t_hi);

    static const std::string rogue_types[] = {"spoof_uid", "erratic_timing", "replay"};
    std::uniform_int_distribution<int> type_pick(0, 2);

    for (int i = 0; i < num_rogue_events; ++i) {
        double      t    = time_dist(rng_);
        double      dur  = 300.0;                     // 5 minutes
        std::string type = rogue_types[type_pick(rng_)];
        rogue_schedule_.emplace_back(t, dur, type);
    }
    // Sort by start time so we can iterate front-to-back
    std::sort(rogue_schedule_.begin(), rogue_schedule_.end());
}

// FIX: renamed from run() to step() to match header declaration
void BeaconSimulator::step(double dt, std::function<void(const Advert&)> callback) {
    if (!isRunning()) return;
    current_time_ += dt;

    // ── Check scheduled rogue injections ─────────────────────────────────────
    // FIX: pass the scheduled duration and type instead of re-randomising
    for (auto it = rogue_schedule_.begin(); it != rogue_schedule_.end(); ) {
        if (std::get<0>(*it) <= current_time_) {
            injectRogue(current_time_, std::get<1>(*it), std::get<2>(*it));
            it = rogue_schedule_.erase(it);
        } else {
            ++it;
        }
    }

    // ── Remove rogues whose time has expired ──────────────────────────────────
    removeExpiredRogues(current_time_);

    // ── Update every device and emit adverts ──────────────────────────────────
    for (auto& dev : devices_) {
        // Mobile devices have non-zero velocity
        if (dev.vx != 0.0 || dev.vy != 0.0)
            dev.updatePosition(dt, width_, height_, rng_);   // FIX: pass rng_

        dev.rotateMac(current_time_, rng_);
        dev.rotateUid(current_time_, rng_);

        if (current_time_ >= dev.last_advertisement + dev.advertisement_interval) {
            Advert adv = dev.generateAdvert(current_time_, 0.0, 0.0);
            callback(adv);
            dev.last_advertisement = current_time_;
        }
    }
}

void BeaconSimulator::injectRogue(double current_time,
                                  double duration,
                                  const std::string& type) {
    // FIX: use the passed 'type' and 'duration' – don't re-randomise
    std::vector<Device*> legit;
    for (auto& d : devices_)
        if (!d.is_rogue) legit.push_back(&d);
    if (legit.empty()) return;

    std::uniform_int_distribution<std::size_t> pick(0, legit.size() - 1);
    Device* target = legit[pick(rng_)];

    std::uniform_real_distribution<double> pos_x(0.0, width_);
    std::uniform_real_distribution<double> pos_y(0.0, height_);

    // Erratic devices advertise/rotate much faster
    double mac_int = (type == "erratic_timing") ?  5.0 : 300.0;
    double uid_int = (type == "erratic_timing") ?  5.0 : 300.0;
    double adv_int = (type == "erratic_timing") ? 0.05 :   0.5;

    std::string rogue_id = "rogue_" + std::to_string(devices_.size());
    Device rogue(rogue_id, target->service_id, true, type,
                 pos_x(rng_), pos_y(rng_),
                 mac_int, uid_int, adv_int);

    rogue.mac = random_mac(rng_);

    // spoof_uid and replay copy the target's UID
    if (type == "spoof_uid" || type == "replay")
        rogue.uid = target->uid;
    else
        rogue.uid = random_uid(rng_);

    // FIX: set end_time properly
    rogue.end_time = current_time + duration;
    devices_.push_back(rogue);
}

void BeaconSimulator::removeExpiredRogues(double current_time) {
    // FIX: safe because end_time is now initialized for ALL devices
    devices_.erase(
        std::remove_if(devices_.begin(), devices_.end(),
            [current_time](const Device& d) {
                return d.is_rogue && current_time > d.end_time;
            }),
        devices_.end());
}
