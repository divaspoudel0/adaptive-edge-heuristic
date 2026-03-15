// FIX: must include engine.h first so extern "C" block wraps our definitions
#include "engine.h"
#include "simulator.h"
#include "learner.h"

#include <unordered_map>
#include <vector>
#include <cmath>
#include <string>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <deque>

// FIX: std::clamp is C++17 – provide C++11 version
template<typename T>
static T clamp11(T val, T lo, T hi) {
    return val < lo ? lo : (val > hi ? hi : val);
}

// ─── Feature vector & similarity ─────────────────────────────────────────────
struct Features {
    double mean_rssi;
    double rssi_std;
    double speed;
};

static double cosine_similarity(const Features& a, const Features& b) {
    double dot   = a.mean_rssi * b.mean_rssi + a.rssi_std * b.rssi_std + a.speed * b.speed;
    double normA = a.mean_rssi * a.mean_rssi + a.rssi_std * a.rssi_std + a.speed * a.speed;
    double normB = b.mean_rssi * b.mean_rssi + b.rssi_std * b.rssi_std + b.speed * b.speed;
    if (normA == 0.0 || normB == 0.0) return 0.0;
    return dot / (std::sqrt(normA) * std::sqrt(normB));
}

// ─── Fingerprint ─────────────────────────────────────────────────────────────
class Fingerprint {
public:
    std::string key;          // "mac:uid:service"
    double      first_seen;
    double      last_seen;
    double      rssi_mean;
    double      rssi_m2;      // Welford running second moment
    int         rssi_count;
    double      speed;
    double      last_x, last_y;
    double      adv_interval_ema;
    int         logical_id;
    bool        anomaly;

    Fingerprint(const std::string& k, double ts,
                double rssi, double x, double y)
        : key(k), first_seen(ts), last_seen(ts),
          rssi_mean(rssi), rssi_m2(0.0), rssi_count(1),
          speed(0.0), last_x(x), last_y(y),
          adv_interval_ema(0.5),
          logical_id(-1), anomaly(false)
    {}

    void update(double ts, double rssi, double x, double y) {
        // Welford's online mean & variance
        ++rssi_count;
        double delta  = rssi - rssi_mean;
        rssi_mean    += delta / rssi_count;
        double delta2 = rssi - rssi_mean;
        rssi_m2      += delta * delta2;

        // Speed: EMA of instantaneous speed
        double dt = ts - last_seen;
        if (dt > 0.0) {
            double dx          = x - last_x;
            double dy          = y - last_y;
            double inst_speed  = std::sqrt(dx * dx + dy * dy) / dt;
            const double alpha = 0.3;
            speed = alpha * inst_speed + (1.0 - alpha) * speed;
        }

        // Advertisement interval EMA
        if (ts > last_seen) {
            double interval    = ts - last_seen;
            const double alpha = 0.1;
            adv_interval_ema = alpha * interval + (1.0 - alpha) * adv_interval_ema;
        }

        last_seen = ts;
        last_x    = x;
        last_y    = y;
    }

    Features getFeatures() const {
        double rssi_std = (rssi_count > 1)
            ? std::sqrt(rssi_m2 / (rssi_count - 1))
            : 0.0;
        Features f;
        f.mean_rssi = rssi_mean;
        f.rssi_std  = rssi_std;
        f.speed     = speed;
        return f;
    }

    bool isAnomalous(double rssi_th, double int_th) const {
        double rssi_std = (rssi_count > 1)
            ? std::sqrt(rssi_m2 / (rssi_count - 1))
            : 0.0;
        if (rssi_std          > rssi_th) return true;
        if (adv_interval_ema  < int_th)  return true;
        return false;
    }
};

// ─── SessionManager ───────────────────────────────────────────────────────────
// Assigns logical IDs across MAC/UID rotations using feature similarity.
class SessionManager {
public:
    SessionManager(double sim_thresh, std::size_t max_candidates)
        : similarity_threshold_(sim_thresh),
          max_recent_(max_candidates),
          next_id_(0)
    {}

    int getLogicalId(Fingerprint& fp,
                     const std::vector<std::string>& recent_keys,
                     const std::unordered_map<std::string, Fingerprint>& fps) {
        if (fp.logical_id != -1) return fp.logical_id;

        Features fp_feat = fp.getFeatures();
        double   best_sim = similarity_threshold_;
        int      best_id  = -1;

        std::size_t checked = 0;
        for (auto it = recent_keys.rbegin();
             it != recent_keys.rend() && checked < max_recent_;
             ++it) {
            if (*it == fp.key) continue;
            auto fit = fps.find(*it);
            if (fit == fps.end())           continue;
            if (fit->second.logical_id < 0) continue;

            double sim = cosine_similarity(fp_feat, fit->second.getFeatures());
            if (sim > best_sim) {
                best_sim = sim;
                best_id  = fit->second.logical_id;
            }
            ++checked;
        }

        fp.logical_id = (best_id != -1) ? best_id : next_id_++;
        return fp.logical_id;
    }

    void setThreshold(double sim_th) { similarity_threshold_ = sim_th; }
    int  logicalCount()        const { return next_id_; }

private:
    double      similarity_threshold_;
    std::size_t max_recent_;
    int         next_id_;
};

// ─── Engine ───────────────────────────────────────────────────────────────────
class Engine {
public:
    explicit Engine(const std::string& config_json) {
        // ── Parse config (simple manual key-value search) ─────────────────────
        // Defaults
        int    num_static    = 10;
        int    num_mobile    =  5;
        double rogue_percent = 10.0;
        double duration_hrs  =  1.0;
        double rssi_th       = 10.0;
        double int_th        =  0.1;
        double sim_th        =  0.8;
        double width         = 100.0;
        double height        = 100.0;

        // Minimal JSON number extraction for each known key
        auto extractDouble = [&](const std::string& key, double def) -> double {
            std::size_t pos = config_json.find("\"" + key + "\"");
            if (pos == std::string::npos) return def;
            pos = config_json.find(':', pos);
            if (pos == std::string::npos) return def;
            try { return std::stod(config_json.substr(pos + 1)); }
            catch (...) { return def; }
        };

        num_static    = static_cast<int>(extractDouble("num_static",    num_static));
        num_mobile    = static_cast<int>(extractDouble("num_mobile",    num_mobile));
        rogue_percent = extractDouble("rogue_percent", rogue_percent);
        duration_hrs  = extractDouble("duration_hours", duration_hrs);
        rssi_th       = extractDouble("rssi_th",  rssi_th);
        int_th        = extractDouble("int_th",   int_th);
        sim_th        = extractDouble("sim_th",   sim_th);
        width         = extractDouble("width",    width);
        height        = extractDouble("height",   height);

        rssi_th_ = rssi_th;
        int_th_  = int_th;
        sim_th_  = sim_th;

        simulator_   = new BeaconSimulator(num_static, num_mobile, rogue_percent,
                                           duration_hrs, width, height);
        learner_     = new ThresholdLearner(rssi_th, int_th, sim_th);
        session_mgr_ = new SessionManager(sim_th, 1000);
    }

    ~Engine() {
        delete simulator_;
        delete learner_;
        delete session_mgr_;
    }

    // Disable copy/assign
    Engine(const Engine&)            = delete;
    Engine& operator=(const Engine&) = delete;

    void setCallback(AdvertCallback cb) { callback_ = cb; }

    void setThresholds(double rssi_th, double int_th, double sim_th) {
        rssi_th_ = rssi_th;
        int_th_  = int_th;
        sim_th_  = sim_th;
        session_mgr_->setThreshold(sim_th);
        learner_->setThresholds(rssi_th, int_th, sim_th);
    }

    int step(double dt) {
        // FIX: simulator now exposes step(), not run()
        simulator_->step(dt, [this](const Advert& adv) {
            processAdvert(adv);
        });

        // Periodic learner update every 100 steps
        if ((++step_counter_ % 100) == 0) {
            learner_->update();
            setThresholds(learner_->rssiTh(),
                          learner_->intTh(),
                          learner_->simTh());
        }

        return simulator_->isRunning() ? 1 : 0;
    }

    const char* getStats() {
        std::ostringstream oss;
        oss << "{"
            << "\"time\":"          << simulator_->currentTime()   << ","
            << "\"logical_count\":" << session_mgr_->logicalCount()<< ","
            << "\"anomaly_rate\":"  << learner_->recentAnomalyRate()<< ","
            << "\"rssi_th\":"       << learner_->rssiTh()          << ","
            << "\"int_th\":"        << learner_->intTh()           << ","
            << "\"sim_th\":"        << learner_->simTh()
            << "}";
        last_stats_ = oss.str();
        return last_stats_.c_str();
    }

private:
    // ── Heuristic processing pipeline ─────────────────────────────────────────
    void processAdvert(const Advert& adv) {
        std::string key = adv.mac + ":" + adv.uid + ":" + adv.service_id;

        // ── Find or create fingerprint ─────────────────────────────────────
        auto it = fingerprints_.find(key);
        if (it == fingerprints_.end()) {
            fingerprints_.emplace(key,
                Fingerprint(key, adv.timestamp, adv.rssi, adv.x, adv.y));
            it = fingerprints_.find(key);
            recent_keys_.push_back(key);
        } else {
            it->second.update(adv.timestamp, adv.rssi, adv.x, adv.y);
        }

        // ── Anomaly detection ──────────────────────────────────────────────
        bool anomaly   = it->second.isAnomalous(rssi_th_, int_th_);
        it->second.anomaly = anomaly;

        // ── Session continuity (logical device tracking) ───────────────────
        int logical_id = session_mgr_->getLogicalId(
            it->second, recent_keys_, fingerprints_);

        // ── Feedback to learner ────────────────────────────────────────────
        learner_->addObservation(adv.is_rogue, anomaly,
                                 adv.rssi, adv.x, adv.y, adv.timestamp);

        // ── Fire Python callback ───────────────────────────────────────────
        if (callback_) {
            callback_(adv.timestamp,
                      adv.mac.c_str(),
                      adv.uid.c_str(),
                      adv.service_id.c_str(),
                      adv.rssi,
                      adv.x,
                      adv.y,
                      adv.is_rogue ? 1 : 0,
                      adv.rogue_type.c_str(),
                      logical_id,
                      anomaly ? 1 : 0);
        }

        // ── Bound the recent-keys list to avoid unbounded growth ───────────
        if (recent_keys_.size() > 20000) {
            recent_keys_.erase(recent_keys_.begin(),
                               recent_keys_.begin() + 10000);
        }
    }

    BeaconSimulator*                             simulator_   = nullptr;
    ThresholdLearner*                            learner_     = nullptr;
    SessionManager*                              session_mgr_ = nullptr;
    AdvertCallback                               callback_    = nullptr;

    std::unordered_map<std::string, Fingerprint> fingerprints_;
    std::vector<std::string>                     recent_keys_;

    double      rssi_th_, int_th_, sim_th_;
    std::string last_stats_;
    int         step_counter_ = 0;
};

// ─── C interface ──────────────────────────────────────────────────────────────
extern "C" {

EngineHandle create_engine(const char* config_json) {
    return new Engine(config_json ? config_json : "{}");
}

void destroy_engine(EngineHandle engine) {
    delete static_cast<Engine*>(engine);
}

void set_callback(EngineHandle engine, AdvertCallback cb) {
    static_cast<Engine*>(engine)->setCallback(cb);
}

int run_step(EngineHandle engine, double dt) {
    return static_cast<Engine*>(engine)->step(dt);
}

const char* get_stats_json(EngineHandle engine) {
    return static_cast<Engine*>(engine)->getStats();
}

void update_thresholds(EngineHandle engine,
                       double rssi_th, double int_th, double sim_th) {
    static_cast<Engine*>(engine)->setThresholds(rssi_th, int_th, sim_th);
}

} // extern "C"
