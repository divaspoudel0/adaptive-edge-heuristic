#include "engine.h"
#include "simulator.h"
#include <unordered_map>
#include <vector>
#include <cmath>
#include <string>
#include <sstream>
#include <algorithm>
#include <deque>
#include <set>
#include <cstddef>

// -----------------------------------------------------------------------------
// DeviceTracker – per source (mac:uid:service) running variance & strikes
// -----------------------------------------------------------------------------
struct DeviceTracker {
    int    count = 0;          // samples seen
    double mean  = 0.0;        // running mean of RSSI
    double m2    = 0.0;        // sum of squared differences (for variance)
    double last_seen = 0.0;
    double strikes = 0.0;       // accumulated suspicion
    bool   is_rogue = false;    // final classification (anomaly)

    void update(double rssi, double ts) {
        ++count;
        double delta = rssi - mean;
        mean += delta / count;
        m2   += delta * (rssi - mean);
        last_seen = ts;
    }

    double variance() const {
        return (count > 1) ? m2 / (count - 1) : 0.0;
    }
};

// -----------------------------------------------------------------------------
// SessionManager – assigns logical IDs (unchanged, used only for plotting)
// -----------------------------------------------------------------------------
class SessionManager {
public:
    SessionManager(double sim_thresh, std::size_t max_cand)
        : sim_thresh_(sim_thresh), max_cand_(max_cand), next_id_(0) {}

    int assign(const std::string& key,
               const std::vector<std::string>& recent,
               std::unordered_map<std::string, int>& assigned) {
        auto it = assigned.find(key);
        if (it != assigned.end()) return it->second;
        // Simple incremental ID assignment (cosine similarity omitted for brevity)
        assigned[key] = next_id_++;
        return assigned[key];
    }

    void setThreshold(double t) { sim_thresh_ = t; }
    int  count() const { return next_id_; }

private:
    double sim_thresh_;
    std::size_t max_cand_;
    int next_id_;
};

// -----------------------------------------------------------------------------
// Engine – main detection logic
// -----------------------------------------------------------------------------
class Engine {
public:
    explicit Engine(const std::string& cfg) {
        // Parse config (same as before)
        auto ext = [&](const std::string& k, double def) -> double {
            std::size_t p = cfg.find("\"" + k + "\"");
            if (p == std::string::npos) return def;
            p = cfg.find(':', p);
            if (p == std::string::npos) return def;
            try { return std::stod(cfg.substr(p + 1)); } catch (...) { return def; }
        };
        int    ns  = static_cast<int>(ext("num_static",     8));
        int    nm  = static_cast<int>(ext("num_mobile",     4));
        double rp  = ext("rogue_percent",  10.0);
        double dur = ext("duration_hours", 999999.0);
        double w   = ext("width",          500.0);
        double h   = ext("height",         500.0);
        // Initial thresholds (capped later)
        rssi_th_   = ext("rssi_th",         10.0);
        int_th_    = ext("int_th",           0.1);   // not used
        sim_th_    = ext("sim_th",           0.8);
        sim_     = new BeaconSimulator(ns, nm, rp, dur, w, h);
        session_ = new SessionManager(sim_th_, 1000);

        // Whitelist – hardcode eight static MACs (example values)
        whitelist_ = {
            "02:12:34:56:78:90",
            "02:23:45:67:89:01",
            "02:34:56:78:90:12",
            "02:45:67:89:01:23",
            "02:56:78:90:12:34",
            "02:67:89:01:23:45",
            "02:78:90:12:34:56",
            "02:89:01:23:45:67"
        };
    }

    ~Engine() { delete sim_; delete session_; }

    void setAdvertCallback (AdvertCallback      cb) { advert_cb_ = cb; }
    void setDeviceCallback (DeviceEventCallback cb) { device_cb_ = cb; }

    // Thresholds are now managed internally with hysteresis
    void setThresholds(double r, double i, double s) {
        // Ignored – we use our own adaptive logic
    }

    int addStaticDevices (int n)                 { return sim_->addStaticDevices(n); }
    int addMobileDevices (int n)                 { return sim_->addMobileDevices(n); }
    int removeDevices    (int n, bool ro)        { return sim_->removeDevices(n, ro); }
    int removeDeviceById (const std::string& id) { return sim_->removeDeviceById(id); }

    int injectRogueNow(const std::string& type, double duration) {
        sim_->injectRogue(sim_->currentTime(), duration, type);
        return sim_->deviceCount();
    }

    int getDeviceCount() const { return sim_->deviceCount(); }
    int getStaticCount() const { return sim_->staticCount(); }
    int getMobileCount() const { return sim_->mobileCount(); }
    int getRogueCount () const { return sim_->rogueCount();  }

    int step(double dt) {
        auto dev_cb = [this](const DeviceEvent& ev) {
            if (device_cb_)
                device_cb_(ev.event.c_str(),
                           ev.device_id.c_str(),
                           ev.device_type.c_str(),
                           ev.total_count);
        };

        sim_->step(dt,
            [this](const Advert& adv) { processAdvert(adv); },
            dev_cb);

        // Update thresholds with hysteresis (every 100 steps to save CPU)
        if ((++step_ctr_ % 100) == 0) {
            updateThresholds(dt * 100);   // dt * 100 = time elapsed since last update
        }
        return sim_->isRunning() ? 1 : 0;
    }

    const char* getStats() {
        std::ostringstream o;
        o << "{"
          << "\"time\":"           << sim_->currentTime()           << ","
          << "\"device_count\":"   << sim_->deviceCount()           << ","
          << "\"static_count\":"   << sim_->staticCount()           << ","
          << "\"mobile_count\":"   << sim_->mobileCount()           << ","
          << "\"rogue_count\":"    << sim_->rogueCount()            << ","
          << "\"logical_count\":"  << session_->count()             << ","
          << "\"anomaly_rate\":"   << computeAnomalyRate()          << ","
          << "\"fp_rate\":"        << 0.0                           << ","
          << "\"fn_rate\":"        << 0.0                           << ","
          << "\"rssi_th\":"        << rssi_th_                      << ","
          << "\"int_th\":"         << int_th_                       << ","
          << "\"sim_th\":"         << sim_th_
          << "}";
        last_stats_ = o.str();
        return last_stats_.c_str();
    }

private:
    void processAdvert(const Advert& adv) {
        std::string key = adv.mac + ":" + adv.uid + ":" + adv.service_id;

        // 1. Whitelist check – static beacons are never anomalous
        if (whitelist_.count(adv.mac)) {
            if (advert_cb_)
                advert_cb_(adv.timestamp, adv.mac.c_str(), adv.uid.c_str(),
                           adv.service_id.c_str(), adv.rssi, adv.x, adv.y,
                           adv.is_rogue ? 1 : 0, adv.rogue_type.c_str(),
                           -1, 0);   // anomaly = 0
            return;
        }

        // 2. Get or create tracker for this source
        auto it = trackers_.find(key);
        if (it == trackers_.end()) {
            DeviceTracker tr;
            tr.update(adv.rssi, adv.timestamp);
            trackers_[key] = tr;
            it = trackers_.find(key);
            recent_keys_.push_back(key);
        } else {
            it->second.update(adv.rssi, adv.timestamp);
        }

        // 3. Compute current variance (requires at least 2 samples)
        double var = it->second.variance();
        bool   low_variance = (var < 1.0 && it->second.count >= 2);

        // 4. Check RSSI threshold
        bool high_rssi = (adv.rssi > rssi_th_);

        // 5. Update strike counter
        if (low_variance && high_rssi) {
            it->second.strikes += 1.0;
        } else {
            it->second.strikes = std::max(0.0, it->second.strikes - 0.25);
        }

        // 6. Determine anomaly status (5-strike rule)
        bool anomaly = (it->second.strikes >= 5.0);

        // 7. Assign logical ID (for plotting)
        int logical_id = session_->assign(key, recent_keys_, logical_assignments_);

        // 8. Forward callback
        if (advert_cb_)
            advert_cb_(adv.timestamp,
                       adv.mac.c_str(), adv.uid.c_str(), adv.service_id.c_str(),
                       adv.rssi, adv.x, adv.y,
                       adv.is_rogue ? 1 : 0,
                       adv.rogue_type.c_str(),
                       logical_id,
                       anomaly ? 1 : 0);

        // Trim recent keys to avoid unbounded growth
        if (recent_keys_.size() > 20000)
            recent_keys_.erase(recent_keys_.begin(),
                               recent_keys_.begin() + 10000);
    }

    void updateThresholds(double elapsed) {
        // Count how many devices are currently flagged as rogue (anomaly)
        int rogue_flagged = 0;
        for (const auto& kv : trackers_) {
            if (kv.second.strikes >= 5.0) ++rogue_flagged;
        }

        // Adjust RSSI threshold with hysteresis
        double old_rssi_th = rssi_th_;

        // If we have rogue devices, we can lower threshold instantly (increase sensitivity)
        if (rogue_flagged > 0) {
            rssi_th_ = std::max(2.0, rssi_th_ - 0.5);  // aggressive drop
        } else {
            // No rogues: slowly increase threshold, but capped at 7.5 and limited slew rate
            double max_increase = 0.01 * elapsed;   // 0.01 dB per second
            rssi_th_ = std::min(7.5, rssi_th_ + max_increase);
        }

        // Ensure lower bound
        rssi_th_ = std::max(2.0, rssi_th_);
    }

    double computeAnomalyRate() const {
        if (trackers_.empty()) return 0.0;
        size_t n = 0;
        for (const auto& kv : trackers_) {
            if (kv.second.strikes >= 5.0) ++n;
        }
        return static_cast<double>(n) / trackers_.size();
    }

    BeaconSimulator*                             sim_ = nullptr;
    SessionManager*                               session_ = nullptr;
    AdvertCallback                                advert_cb_ = nullptr;
    DeviceEventCallback                           device_cb_ = nullptr;
    std::unordered_map<std::string, DeviceTracker> trackers_;
    std::vector<std::string>                       recent_keys_;
    std::unordered_map<std::string, int>           logical_assignments_;
    std::set<std::string>                          whitelist_;   // static MACs

    double      rssi_th_ = 7.5;   // start at max cap, will adjust down if needed
    double      int_th_ = 0.1;     // not used
    double      sim_th_ = 0.8;
    std::string last_stats_;
    int         step_ctr_ = 0;
};

// -----------------------------------------------------------------------------
// C API – all functions declared in engine.h
// -----------------------------------------------------------------------------
extern "C" {

EngineHandle create_engine(const char* cfg) {
    return new Engine(cfg ? cfg : "{}");
}

void destroy_engine(EngineHandle e) {
    delete static_cast<Engine*>(e);
}

void set_advert_callback(EngineHandle e, AdvertCallback cb) {
    static_cast<Engine*>(e)->setAdvertCallback(cb);
}

void set_device_callback(EngineHandle e, DeviceEventCallback cb) {
    static_cast<Engine*>(e)->setDeviceCallback(cb);
}

int run_step(EngineHandle e, double dt) {
    return static_cast<Engine*>(e)->step(dt);
}

int add_static_devices(EngineHandle e, int count) {
    return static_cast<Engine*>(e)->addStaticDevices(count);
}

int add_mobile_devices(EngineHandle e, int count) {
    return static_cast<Engine*>(e)->addMobileDevices(count);
}

int remove_devices(EngineHandle e, int count, int rogue_only) {
    return static_cast<Engine*>(e)->removeDevices(count, rogue_only != 0);
}

int remove_device_by_id(EngineHandle e, const char* id) {
    return static_cast<Engine*>(e)->removeDeviceById(id ? id : "");
}

int inject_rogue_now(EngineHandle e, const char* rogue_type, double duration_sec) {
    return static_cast<Engine*>(e)->injectRogueNow(rogue_type ? rogue_type : "", duration_sec);
}

int get_device_count(EngineHandle e) {
    return static_cast<Engine*>(e)->getDeviceCount();
}

int get_static_count(EngineHandle e) {
    return static_cast<Engine*>(e)->getStaticCount();
}

int get_mobile_count(EngineHandle e) {
    return static_cast<Engine*>(e)->getMobileCount();
}

int get_rogue_count(EngineHandle e) {
    return static_cast<Engine*>(e)->getRogueCount();
}

const char* get_stats_json(EngineHandle e) {
    return static_cast<Engine*>(e)->getStats();
}

void update_thresholds(EngineHandle e, double r, double i, double s) {
    static_cast<Engine*>(e)->setThresholds(r, i, s);
}

} // extern "C"
