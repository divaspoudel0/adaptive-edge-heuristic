#include "simulator.h"
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <tuple>
#include <cstddef>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ─── C++11 compatible clamp ───────────────────────────────────────────────────
template<typename T>
static T clamp11(T val, T lo, T hi) {
    return val < lo ? lo : (val > hi ? hi : val);
}

// ─── RSSI path-loss model ─────────────────────────────────────────────────────
static double rssi_from_distance(double d, double P0 = -59.0, double n = 2.0) {
    if (d < 0.01) d = 0.01;
    return P0 - 10.0 * n * std::log10(d);
}

// ─── Random identifier helpers ────────────────────────────────────────────────
static std::string random_mac(std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(0, 255);
    std::ostringstream ss;
    ss << "02";
    for (int i = 0; i < 5; ++i)
        ss << ":" << std::hex << std::setfill('0') << std::setw(2) << dist(rng);
    return ss.str();
}

static std::string random_uid(std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(0, 255);
    std::ostringstream ss;
    for (int g = 0; g < 5; ++g) {
        if (g > 0) ss << "-";
        int len = (g == 0) ? 4 : (g < 4) ? 2 : 6;
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
               double mac_int, double uid_int,
               double adv_int,
               bool mobile)
    : device_id(id), service_id(svc),
      is_rogue(rogue), rogue_type(rtype),
      x(x0), y(y0), vx(0.0), vy(0.0),
      mac_rotation_interval(mac_int),
      uid_rotation_interval(uid_int),
      advertisement_interval(adv_int),
      last_advertisement(-adv_int),   // fire on first step
      next_mac_rotation(mac_int),
      next_uid_rotation(uid_int),
      end_time(1e18),                 // permanent by default
      mac(""), uid(""),
      is_mobile(mobile)
{}

void Device::updatePosition(double dt, double width, double height, std::mt19937& rng) {
    // ~1% chance per second to change direction
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    if (prob(rng) < 0.01 * dt) {
        std::uniform_real_distribution<double> angle_d(0.0, 2.0 * M_PI);
        std::uniform_real_distribution<double> speed_d(0.5, 2.0);
        double angle = angle_d(rng);
        double speed = speed_d(rng);
        vx = speed * std::cos(angle);
        vy = speed * std::sin(angle);
    }
    x += vx * dt;
    y += vy * dt;
    // Elastic bounce off walls
    if (x < 0.0)    { vx =  std::abs(vx); x = 0.0;    }
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
    adv.logical_id = -1;
    adv.anomaly    = 0;
    double dx = x - rx, dy = y - ry;
    adv.rssi = rssi_from_distance(std::sqrt(dx * dx + dy * dy));
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
      device_serial_(0),
      rng_(std::random_device{}())
{
    // Seed initial static and mobile devices
    for (int i = 0; i < num_static; ++i) addStaticDevices(1);
    for (int i = 0; i < num_mobile; ++i) addMobileDevices(1);
    pending_events_.clear(); // don't fire "added" events before sim starts

    // Schedule rogue injections
    int total          = num_static + num_mobile;
    int num_rogues     = std::max(1, static_cast<int>(total * rogue_percent / 100.0));
    double t_lo        = total_duration_ * 0.05;
    double t_hi        = total_duration_ * 0.95;
    std::uniform_real_distribution<double> t_dist(t_lo, t_hi);
    static const std::string types[] = {"spoof_uid", "erratic_timing", "replay"};
    std::uniform_int_distribution<int> type_pick(0, 2);
    for (int i = 0; i < num_rogues; ++i)
        rogue_schedule_.emplace_back(t_dist(rng_), 300.0, types[type_pick(rng_)]);
    std::sort(rogue_schedule_.begin(), rogue_schedule_.end());
}

// ─── Device factory helpers ───────────────────────────────────────────────────
Device BeaconSimulator::makeStaticDevice(const std::string& id) {
    std::uniform_real_distribution<double> px(0.0, width_);
    std::uniform_real_distribution<double> py(0.0, height_);
    std::uniform_int_distribution<int>     svc(1, 3);
    std::string service = "SVC_" + std::to_string(svc(rng_));
    Device dev(id, service, false, "", px(rng_), py(rng_),
               1e18, 1e18, 0.5, false);
    dev.mac = random_mac(rng_);
    dev.uid = random_uid(rng_);
    return dev;
}

Device BeaconSimulator::makeMobileDevice(const std::string& id) {
    std::uniform_real_distribution<double> px(0.0, width_);
    std::uniform_real_distribution<double> py(0.0, height_);
    std::uniform_real_distribution<double> speed_d(0.5, 2.0);
    std::uniform_real_distribution<double> angle_d(0.0, 2.0 * M_PI);
    std::uniform_real_distribution<double> mac_int_d(120.0, 300.0);
    std::uniform_real_distribution<double> uid_int_d(60.0, 180.0);
    std::uniform_int_distribution<int>     svc(1, 3);
    double angle = angle_d(rng_), speed = speed_d(rng_);
    std::string service = "SVC_" + std::to_string(svc(rng_));
    Device dev(id, service, false, "", px(rng_), py(rng_),
               mac_int_d(rng_), uid_int_d(rng_), 0.5, true);
    dev.vx  = speed * std::cos(angle);
    dev.vy  = speed * std::sin(angle);
    dev.mac = random_mac(rng_);
    dev.uid = random_uid(rng_);
    return dev;
}

void BeaconSimulator::emitEvent(const std::string& event,
                                const std::string& device_id,
                                const std::string& device_type) {
    DeviceEvent ev;
    ev.event       = event;
    ev.device_id   = device_id;
    ev.device_type = device_type;
    ev.total_count = static_cast<int>(devices_.size());
    pending_events_.push_back(ev);
}

// ─── Dynamic add / remove ─────────────────────────────────────────────────────
int BeaconSimulator::addStaticDevices(int count) {
    for (int i = 0; i < count; ++i) {
        std::string id = "static_" + std::to_string(device_serial_++);
        devices_.push_back(makeStaticDevice(id));
        emitEvent("added", id, "static");
    }
    return static_cast<int>(devices_.size());
}

int BeaconSimulator::addMobileDevices(int count) {
    for (int i = 0; i < count; ++i) {
        std::string id = "mobile_" + std::to_string(device_serial_++);
        devices_.push_back(makeMobileDevice(id));
        emitEvent("added", id, "mobile");
    }
    return static_cast<int>(devices_.size());
}

int BeaconSimulator::removeDevices(int count, bool rogue_only) {
    // Remove from the back to avoid shifting issues; skip static if rogue_only
    int removed = 0;
    for (auto it = devices_.end(); it != devices_.begin() && removed < count; ) {
        --it;
        if (rogue_only && !it->is_rogue) continue;
        if (!rogue_only && !it->is_rogue && !it->is_mobile) continue; // keep static
        std::string type = it->is_rogue  ? "rogue"
                         : it->is_mobile ? "mobile" : "static";
        emitEvent("removed", it->device_id, type);
        it = devices_.erase(it);
        ++removed;
    }
    return static_cast<int>(devices_.size());
}

int BeaconSimulator::removeDeviceById(const std::string& id) {
    for (auto it = devices_.begin(); it != devices_.end(); ++it) {
        if (it->device_id == id) {
            std::string type = it->is_rogue  ? "rogue"
                             : it->is_mobile ? "mobile" : "static";
            emitEvent("removed", id, type);
            devices_.erase(it);
            return 1;
        }
    }
    return 0;
}

// ─── Queries ──────────────────────────────────────────────────────────────────
int BeaconSimulator::deviceCount() const {
    return static_cast<int>(devices_.size());
}
int BeaconSimulator::staticCount() const {
    int n = 0;
    for (const auto& d : devices_) if (!d.is_rogue && !d.is_mobile) ++n;
    return n;
}
int BeaconSimulator::mobileCount() const {
    int n = 0;
    for (const auto& d : devices_) if (!d.is_rogue && d.is_mobile) ++n;
    return n;
}
int BeaconSimulator::rogueCount() const {
    int n = 0;
    for (const auto& d : devices_) if (d.is_rogue) ++n;
    return n;
}

// ─── Rogue injection helpers ──────────────────────────────────────────────────
void BeaconSimulator::injectRogue(double current_time,
                                  double duration,
                                  const std::string& type) {
    std::vector<Device*> legit;
    for (auto& d : devices_) if (!d.is_rogue) legit.push_back(&d);
    if (legit.empty()) return;

    std::uniform_int_distribution<std::size_t> pick(0, legit.size() - 1);
    Device* target = legit[pick(rng_)];

    std::uniform_real_distribution<double> px(0.0, width_);
    std::uniform_real_distribution<double> py(0.0, height_);

    double mac_int = (type == "erratic_timing") ?  5.0 : 300.0;
    double uid_int = (type == "erratic_timing") ?  5.0 : 300.0;
    double adv_int = (type == "erratic_timing") ? 0.05 :   0.5;

    std::string id = "rogue_" + std::to_string(device_serial_++);
    Device rogue(id, target->service_id, true, type,
                 px(rng_), py(rng_),
                 mac_int, uid_int, adv_int, false);

    rogue.mac = random_mac(rng_);
    rogue.uid = (type == "spoof_uid" || type == "replay")
              ? target->uid
              : random_uid(rng_);
    rogue.end_time = current_time + duration;
    devices_.push_back(rogue);
    emitEvent("added", id, "rogue");
}

void BeaconSimulator::removeExpiredRogues(double current_time) {
    auto it = devices_.begin();
    while (it != devices_.end()) {
        if (it->is_rogue && current_time > it->end_time) {
            emitEvent("removed", it->device_id, "rogue");
            it = devices_.erase(it);
        } else {
            ++it;
        }
    }
}

// ─── Main simulation step ─────────────────────────────────────────────────────
void BeaconSimulator::step(double dt,
                           std::function<void(const Advert&)>      advert_cb,
                           std::function<void(const DeviceEvent&)> device_cb) {
    if (!isRunning()) return;
    current_time_ += dt;

    // Fire any pending device events first (from add/remove calls between steps)
    for (auto& ev : pending_events_) {
        ev.total_count = static_cast<int>(devices_.size());
        if (device_cb) device_cb(ev);
    }
    pending_events_.clear();

    // Scheduled rogue injections
    for (auto it = rogue_schedule_.begin(); it != rogue_schedule_.end(); ) {
        if (std::get<0>(*it) <= current_time_) {
            injectRogue(current_time_, std::get<1>(*it), std::get<2>(*it));
            it = rogue_schedule_.erase(it);
        } else { ++it; }
    }

    // Fire device events generated by rogue injection
    for (auto& ev : pending_events_) {
        ev.total_count = static_cast<int>(devices_.size());
        if (device_cb) device_cb(ev);
    }
    pending_events_.clear();

    // Expire old rogues
    removeExpiredRogues(current_time_);
    for (auto& ev : pending_events_) {
        ev.total_count = static_cast<int>(devices_.size());
        if (device_cb) device_cb(ev);
    }
    pending_events_.clear();

    // Update devices and emit adverts
    for (auto& dev : devices_) {
        if (dev.is_mobile)
            dev.updatePosition(dt, width_, height_, rng_);
        dev.rotateMac(current_time_, rng_);
        dev.rotateUid(current_time_, rng_);
        if (current_time_ >= dev.last_advertisement + dev.advertisement_interval) {
            Advert adv = dev.generateAdvert(current_time_, 0.0, 0.0);
            if (advert_cb) advert_cb(adv);
            dev.last_advertisement = current_time_;
        }
    }
}
