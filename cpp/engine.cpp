#include "engine.h"
#include "simulator.h"
#include "learner.h"
#include <unordered_map>
#include <vector>
#include <cmath>
#include <string>
#include <sstream>
#include <algorithm>
#include <deque>

template<typename T> static T clamp11(T v, T lo, T hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

struct Features { double mean_rssi, rssi_std, speed; };

static double cosine_sim(const Features& a, const Features& b) {
    double dot   = a.mean_rssi*b.mean_rssi + a.rssi_std*b.rssi_std + a.speed*b.speed;
    double normA = a.mean_rssi*a.mean_rssi + a.rssi_std*a.rssi_std + a.speed*a.speed;
    double normB = b.mean_rssi*b.mean_rssi + b.rssi_std*b.rssi_std + b.speed*b.speed;
    if (normA == 0.0 || normB == 0.0) return 0.0;
    return dot / (std::sqrt(normA) * std::sqrt(normB));
}

class Fingerprint {
public:
    std::string key;
    double first_seen, last_seen;
    double rssi_mean, rssi_m2;
    int    rssi_count;
    double speed, last_x, last_y, adv_interval_ema;
    int    logical_id;
    bool   anomaly;

    Fingerprint(const std::string& k, double ts, double rssi, double x, double y)
        : key(k), first_seen(ts), last_seen(ts),
          rssi_mean(rssi), rssi_m2(0.0), rssi_count(1),
          speed(0.0), last_x(x), last_y(y), adv_interval_ema(0.5),
          logical_id(-1), anomaly(false) {}

    void update(double ts, double rssi, double x, double y) {
        ++rssi_count;
        double d = rssi - rssi_mean; rssi_mean += d / rssi_count;
        rssi_m2 += d * (rssi - rssi_mean);
        double dt = ts - last_seen;
        if (dt > 0.0) {
            double dx = x-last_x, dy = y-last_y;
            speed = 0.3*(std::sqrt(dx*dx+dy*dy)/dt) + 0.7*speed;
            adv_interval_ema = 0.1*dt + 0.9*adv_interval_ema;
        }
        last_seen = ts; last_x = x; last_y = y;
    }

    Features getFeatures() const {
        double s = (rssi_count > 1) ? std::sqrt(rssi_m2/(rssi_count-1)) : 0.0;
        return {rssi_mean, s, speed};
    }

    bool isAnomalous(double rssi_th, double int_th) const {
        double s = (rssi_count > 1) ? std::sqrt(rssi_m2/(rssi_count-1)) : 0.0;
        return (s > rssi_th) || (adv_interval_ema < int_th);
    }
};

class SessionManager {
public:
    SessionManager(double t, std::size_t m) : thresh_(t), max_(m), next_(0) {}
    int assign(Fingerprint& fp,
               const std::vector<std::string>& recent,
               const std::unordered_map<std::string, Fingerprint>& fps) {
        if (fp.logical_id != -1) return fp.logical_id;
        Features f = fp.getFeatures();
        double best = thresh_; int best_id = -1; std::size_t c = 0;
        for (auto it = recent.rbegin(); it != recent.rend() && c < max_; ++it) {
            if (*it == fp.key) continue;
            auto fit = fps.find(*it);
            if (fit == fps.end() || fit->second.logical_id < 0) continue;
            double s = cosine_sim(f, fit->second.getFeatures());
            if (s > best) { best = s; best_id = fit->second.logical_id; }
            ++c;
        }
        fp.logical_id = (best_id != -1) ? best_id : next_++;
        return fp.logical_id;
    }
    void setThreshold(double t) { thresh_ = t; }
    int  count() const { return next_; }
private:
    double thresh_; std::size_t max_; int next_;
};

class Engine {
public:
    explicit Engine(const std::string& cfg) {
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
        double dur = ext("duration_hours", 999999.0); // indefinite by default
        double w   = ext("width",          500.0);
        double h   = ext("height",         500.0);
        rssi_th_   = ext("rssi_th",         10.0);
        int_th_    = ext("int_th",           0.1);
        sim_th_    = ext("sim_th",           0.8);

        sim_     = new BeaconSimulator(ns, nm, rp, dur, w, h);
        learner_ = new ThresholdLearner(rssi_th_, int_th_, sim_th_);
        session_ = new SessionManager(sim_th_, 1000);
    }
    ~Engine() { delete sim_; delete learner_; delete session_; }
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    void setAdvertCallback (AdvertCallback      cb) { advert_cb_  = cb; }
    void setDeviceCallback (DeviceEventCallback cb) { device_cb_  = cb; }

    void setThresholds(double r, double i, double s) {
        rssi_th_ = r; int_th_ = i; sim_th_ = s;
        session_->setThreshold(s); learner_->setThresholds(r, i, s);
    }

    int addStaticDevices (int n)                    { return sim_->addStaticDevices(n); }
    int addMobileDevices (int n)                    { return sim_->addMobileDevices(n); }
    int removeDevices    (int n, bool ro)           { return sim_->removeDevices(n, ro); }
    int removeDeviceById (const std::string& id)    { return sim_->removeDeviceById(id); }

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
                device_cb_(ev.event.c_str(), ev.device_id.c_str(),
                           ev.device_type.c_str(), ev.total_count);
        };
        sim_->step(dt, [this](const Advert& a){ processAdvert(a); }, dev_cb);
        if ((++step_ctr_ % 100) == 0) {
            learner_->update();
            setThresholds(learner_->rssiTh(), learner_->intTh(), learner_->simTh());
        }
        return 1; // always running
    }

    const char* getStats() {
        std::ostringstream o;
        o << "{"
          << "\"time\":"           << sim_->currentTime()          << ","
          << "\"device_count\":"   << sim_->deviceCount()          << ","
          << "\"static_count\":"   << sim_->staticCount()          << ","
          << "\"mobile_count\":"   << sim_->mobileCount()          << ","
          << "\"rogue_count\":"    << sim_->rogueCount()           << ","
          << "\"logical_count\":"  << session_->count()            << ","
          << "\"anomaly_rate\":"   << learner_->recentAnomalyRate() << ","
          << "\"fp_rate\":"        << learner_->recentFPRate()     << ","
          << "\"fn_rate\":"        << learner_->recentFNRate()     << ","
          << "\"rssi_th\":"        << learner_->rssiTh()           << ","
          << "\"int_th\":"         << learner_->intTh()            << ","
          << "\"sim_th\":"         << learner_->simTh()
          << "}";
        last_stats_ = o.str();
        return last_stats_.c_str();
    }

private:
    void processAdvert(const Advert& adv) {
        std::string key = adv.mac + ":" + adv.uid + ":" + adv.service_id;
        auto it = fingerprints_.find(key);
        if (it == fingerprints_.end()) {
            fingerprints_.emplace(key, Fingerprint(key, adv.timestamp, adv.rssi, adv.x, adv.y));
            it = fingerprints_.find(key);
            recent_keys_.push_back(key);
        } else {
            it->second.update(adv.timestamp, adv.rssi, adv.x, adv.y);
        }
        bool anomaly = it->second.isAnomalous(rssi_th_, int_th_);
        it->second.anomaly = anomaly;
        int  lid = session_->assign(it->second, recent_keys_, fingerprints_);
        learner_->addObservation(adv.is_rogue, anomaly,
                                 adv.rssi, adv.x, adv.y, adv.timestamp);
        if (advert_cb_)
            advert_cb_(adv.timestamp,
                       adv.mac.c_str(), adv.uid.c_str(), adv.service_id.c_str(),
                       adv.rssi, adv.x, adv.y,
                       adv.is_rogue ? 1 : 0, adv.rogue_type.c_str(),
                       lid, anomaly ? 1 : 0);
        if (recent_keys_.size() > 20000)
            recent_keys_.erase(recent_keys_.begin(), recent_keys_.begin() + 10000);
    }

    BeaconSimulator*                             sim_      = nullptr;
    ThresholdLearner*                            learner_  = nullptr;
    SessionManager*                              session_  = nullptr;
    AdvertCallback                               advert_cb_  = nullptr;
    DeviceEventCallback                          device_cb_  = nullptr;
    std::unordered_map<std::string, Fingerprint> fingerprints_;
    std::vector<std::string>                     recent_keys_;
    double rssi_th_, int_th_, sim_th_;
    std::string last_stats_;
    int step_ctr_ = 0;
};

// ─── C interface ──────────────────────────────────────────────────────────────
extern "C" {
EngineHandle create_engine(const char* cfg) { return new Engine(cfg ? cfg : "{}"); }
void destroy_engine(EngineHandle e) { delete static_cast<Engine*>(e); }
void set_advert_callback(EngineHandle e, AdvertCallback cb)      { static_cast<Engine*>(e)->setAdvertCallback(cb); }
void set_device_callback(EngineHandle e, DeviceEventCallback cb) { static_cast<Engine*>(e)->setDeviceCallback(cb); }
int  run_step(EngineHandle e, double dt)                         { return static_cast<Engine*>(e)->step(dt); }
int  add_static_devices(EngineHandle e, int n)                   { return static_cast<Engine*>(e)->addStaticDevices(n); }
int  add_mobile_devices(EngineHandle e, int n)                   { return static_cast<Engine*>(e)->addMobileDevices(n); }
int  remove_devices(EngineHandle e, int n, int ro)               { return static_cast<Engine*>(e)->removeDevices(n, ro != 0); }
int  remove_device_by_id(EngineHandle e, const char* id)         { return static_cast<Engine*>(e)->removeDeviceById(id ? id : ""); }
int  inject_rogue_now(EngineHandle e, const char* type, double dur) {
    static const std::string valid[] = {"spoof_uid","erratic_timing","replay"};
    std::string t = (type ? type : "spoof_uid");
    bool ok = false;
    for (const auto& v : valid) if (v == t) { ok = true; break; }
    if (!ok) t = "spoof_uid";
    return static_cast<Engine*>(e)->injectRogueNow(t, dur);
}
int         get_device_count(EngineHandle e) { return static_cast<Engine*>(e)->getDeviceCount(); }
int         get_static_count(EngineHandle e) { return static_cast<Engine*>(e)->getStaticCount(); }
int         get_mobile_count(EngineHandle e) { return static_cast<Engine*>(e)->getMobileCount(); }
int         get_rogue_count (EngineHandle e) { return static_cast<Engine*>(e)->getRogueCount();  }
const char* get_stats_json  (EngineHandle e) { return static_cast<Engine*>(e)->getStats(); }
void update_thresholds(EngineHandle e, double r, double i, double s) { static_cast<Engine*>(e)->setThresholds(r, i, s); }
}
