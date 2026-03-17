#include "engine.h"
#include "simulator.h"
#include "learner.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <deque>
#include <cmath>
#include <string>
#include <sstream>
#include <algorithm>
#include <cstddef>
#include <cstdint>

// --- Constants for "Strict Zero" ---
const size_t VARIANCE_WINDOW = 15;          // N = 15 samples
const double VARIANCE_LIMIT  = 1.25;         // σ < 1.25 → too steady → rogue-like
const int    STRIKE_THRESHOLD = 3;           // need 3 strikes to declare rogue
const double STRIKE_DECAY     = 0.25;         // subtract this when packet passes

template<typename T>
static T clamp11(T val, T lo, T hi) {
    return val < lo ? lo : (val > hi ? hi : val);
}

// --- Feature vector (unchanged) ---
struct Features {
    double mean_rssi;
    double rssi_std;
    double speed;
};

static double cosine_similarity(const Features& a, const Features& b) {
    double dot   = a.mean_rssi*b.mean_rssi + a.rssi_std*b.rssi_std + a.speed*b.speed;
    double normA = a.mean_rssi*a.mean_rssi + a.rssi_std*a.rssi_std + a.speed*a.speed;
    double normB = b.mean_rssi*b.mean_rssi + b.rssi_std*b.rssi_std + b.speed*b.speed;
    if (normA == 0.0 || normB == 0.0) return 0.0;
    return dot / (std::sqrt(normA) * std::sqrt(normB));
}

// --- Fingerprint (now with variance history and suspicion) ---
class Fingerprint {
public:
    std::string key;
    double      first_seen, last_seen;
    double      rssi_mean, rssi_m2;
    int         rssi_count;
    double      speed;
    double      last_x, last_y;
    double      adv_interval_ema;
    int         logical_id;
    bool        anomaly;               // final decision after strike counter

    // New fields for "Strict Zero"
    std::deque<double> rssi_history;    // last N RSSI values for variance
    double             suspicion;        // current strike count (0..)
    // Optional: bool is_static; // set from device event if we had device_type

    Fingerprint(const std::string& k, double ts, double rssi, double x, double y)
        : key(k), first_seen(ts), last_seen(ts),
          rssi_mean(rssi), rssi_m2(0.0), rssi_count(1),
          speed(0.0), last_x(x), last_y(y),
          adv_interval_ema(0.5),
          logical_id(-1), anomaly(false),
          suspicion(0.0)
    {
        rssi_history.push_back(rssi);
    }

    void update(double ts, double rssi, double x, double y) {
        ++rssi_count;
        double delta  = rssi - rssi_mean;
        rssi_mean    += delta / rssi_count;
        rssi_m2      += delta * (rssi - rssi_mean);

        double dt = ts - last_seen;
        if (dt > 0.0) {
            double dx = x - last_x, dy = y - last_y;
            speed = 0.3 * (std::sqrt(dx*dx + dy*dy) / dt) + 0.7 * speed;
        }
        if (ts > last_seen)
            adv_interval_ema = 0.1*(ts-last_seen) + 0.9*adv_interval_ema;

        last_seen = ts; last_x = x; last_y = y;

        // Update RSSI history for variance
        rssi_history.push_back(rssi);
        if (rssi_history.size() > VARIANCE_WINDOW)
            rssi_history.pop_front();
    }

    // Calculate variance over the current history window
    double currentVariance() const {
        if (rssi_history.size() < 2) return 0.0;
        double sum = 0.0, sum2 = 0.0;
        for (double v : rssi_history) {
            sum += v;
            sum2 += v * v;
        }
        size_t n = rssi_history.size();
        double mean = sum / n;
        return (sum2 / n) - (mean * mean);
    }

    Features getFeatures() const {
        double std = (rssi_count > 1)
            ? std::sqrt(rssi_m2 / (rssi_count - 1)) : 0.0;
        return {rssi_mean, std, speed};
    }

    // Legacy simple heuristic (still used, but combined with variance)
    bool passesSimpleHeuristic(double rssi_th, double int_th) const {
        double std = (rssi_count > 1)
            ? std::sqrt(rssi_m2 / (rssi_count - 1)) : 0.0;
        return (std <= rssi_th) && (adv_interval_ema >= int_th);
    }
};

// --- SessionManager (unchanged) ---
class SessionManager {
public:
    SessionManager(double sim_thresh, std::size_t max_cand)
        : sim_thresh_(sim_thresh), max_cand_(max_cand), next_id_(0) {}

    int assign(Fingerprint& fp,
               const std::vector<std::string>& recent,
               const std::unordered_map<std::string, Fingerprint>& fps) {
        if (fp.logical_id != -1) return fp.logical_id;
        Features feat = fp.getFeatures();
        double best_sim = sim_thresh_;
        int best_id = -1;
        std::size_t checked = 0;
        for (auto it = recent.rbegin();
             it != recent.rend() && checked < max_cand_; ++it) {
            if (*it == fp.key) continue;
            auto fit = fps.find(*it);
            if (fit == fps.end() || fit->second.logical_id < 0) continue;
            double s = cosine_similarity(feat, fit->second.getFeatures());
            if (s > best_sim) { best_sim = s; best_id = fit->second.logical_id; }
            ++checked;
        }
        fp.logical_id = (best_id != -1) ? best_id : next_id_++;
        return fp.logical_id;
    }

    void setThreshold(double t) { sim_thresh_ = t; }
    int  count() const { return next_id_; }

private:
    double      sim_thresh_;
    std::size_t max_cand_;
    int         next_id_;
};

// --- Engine ---
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
        double dur = ext("duration_hours", 999999.0);
        double w   = ext("width",          500.0);
        double h   = ext("height",         500.0);
        rssi_th_   = ext("rssi_th",         6.5);    // lower initial cap
        int_th_    = ext("int_th",           0.2);   // slightly higher
        sim_th_    = ext("sim_th",           0.8);
        sim_     = new BeaconSimulator(ns, nm, rp, dur, w, h);
        learner_ = new ThresholdLearner(rssi_th_, int_th_, sim_th_);
        session_ = new SessionManager(sim_th_, 1000);
    }

    ~Engine() { delete sim_; delete learner_; delete session_; }
    Engine(const Engine&)            = delete;
    Engine& operator=(const Engine&) = delete;

    void setAdvertCallback (AdvertCallback      cb) { advert_cb_ = cb; }
    void setDeviceCallback (DeviceEventCallback cb) { device_cb_ = cb; }

    void setThresholds(double r, double i, double s) {
        rssi_th_ = r; int_th_ = i; sim_th_ = s;
        session_->setThreshold(s);
        learner_->setThresholds(r, i, s);
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
            // If we had device_type in advert, we could record static MACs here
        };

        sim_->step(dt,
            [this](const Advert& adv) { processAdvert(adv); },
            dev_cb);

        if ((++step_ctr_ % 50) == 0) {   // update every 50 steps (5s at dt=0.1)
            learner_->update();
            // Apply hysteresis: only allow slow increase, fast decrease
            double new_rssi = learner_->rssiTh();
            double new_int  = learner_->intTh();
            double new_sim  = learner_->simTh();

            // Hysteresis: RSSI threshold can increase at most 0.1 per update (0.02/s)
            // but can decrease arbitrarily.
            const double MAX_INCREASE_PER_UPDATE = 0.1;   // 5s * 0.02/s
            if (new_rssi > rssi_th_) {
                rssi_th_ = std::min(new_rssi, rssi_th_ + MAX_INCREASE_PER_UPDATE);
            } else {
                rssi_th_ = new_rssi;   // drop immediately
            }

            // Interval threshold: similar logic
            const double MAX_INT_INCREASE_PER_UPDATE = 0.02;
            if (new_int > int_th_) {
                int_th_ = std::min(new_int, int_th_ + MAX_INT_INCREASE_PER_UPDATE);
            } else {
                int_th_ = new_int;
            }

            // Similar for sim_th (optional)
            if (new_sim > sim_th_) {
                sim_th_ = std::min(new_sim, sim_th_ + 0.02);
            } else {
                sim_th_ = new_sim;
            }

            // Cap RSSI at 6.5 to prevent blindness
            rssi_th_ = std::min(rssi_th_, 6.5);
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
          << "\"anomaly_rate\":"   << learner_->recentAnomalyRate() << ","
          << "\"fp_rate\":"        << learner_->recentFPRate()      << ","
          << "\"fn_rate\":"        << learner_->recentFNRate()      << ","
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

        auto it = fingerprints_.find(key);
        if (it == fingerprints_.end()) {
            fingerprints_.emplace(key,
                Fingerprint(key, adv.timestamp, adv.rssi, adv.x, adv.y));
            it = fingerprints_.find(key);
            recent_keys_.push_back(key);
        } else {
            it->second.update(adv.timestamp, adv.rssi, adv.x, adv.y);
        }

        // --- Step 1: Check simple heuristics (RSSI std & interval) ---
        bool simple_pass = it->second.passesSimpleHeuristic(rssi_th_, int_th_);

        // --- Step 2: Variance check (fingerprinting) ---
        double var = it->second.currentVariance();
        bool low_variance = (var < VARIANCE_LIMIT);

        // --- Step 3: Update suspicion counter ---
        // Suspicious if either simple fails or variance is too low
        bool suspicious = !simple_pass || low_variance;
        if (suspicious) {
            it->second.suspicion += 1.0;
        } else {
            it->second.suspicion -= STRIKE_DECAY;
        }
        // Clamp suspicion between 0 and a high value (e.g., 10)
        if (it->second.suspicion < 0.0) it->second.suspicion = 0.0;
        if (it->second.suspicion > 10.0) it->second.suspicion = 10.0; // optional

        // --- Step 4: Final anomaly decision (3‑strike rule) ---
        bool anomaly = (it->second.suspicion >= STRIKE_THRESHOLD);
        it->second.anomaly = anomaly;

        // Logical ID assignment (unchanged)
        int logical_id = session_->assign(it->second, recent_keys_, fingerprints_);

        // Feed observation to learner (ground truth from simulator)
        learner_->addObservation(adv.is_rogue, anomaly,
                                 adv.rssi, adv.x, adv.y, adv.timestamp);

        // Callback (unchanged signature)
        if (advert_cb_)
            advert_cb_(adv.timestamp,
                       adv.mac.c_str(), adv.uid.c_str(), adv.service_id.c_str(),
                       adv.rssi, adv.x, adv.y,
                       adv.is_rogue ? 1 : 0,
                       adv.rogue_type.c_str(),
                       logical_id, anomaly ? 1 : 0);

        // Limit recent_keys size
        if (recent_keys_.size() > 20000)
            recent_keys_.erase(recent_keys_.begin(),
                               recent_keys_.begin() + 10000);
    }

    BeaconSimulator*                             sim_       = nullptr;
    ThresholdLearner*                            learner_   = nullptr;
    SessionManager*                              session_   = nullptr;
    AdvertCallback                               advert_cb_ = nullptr;
    DeviceEventCallback                          device_cb_ = nullptr;
    std::unordered_map<std::string, Fingerprint> fingerprints_;
    std::vector<std::string>                     recent_keys_;

    double      rssi_th_, int_th_, sim_th_;
    std::string last_stats_;
    int         step_ctr_ = 0;
};

extern "C" {
    // ... (same as before) ...
}
