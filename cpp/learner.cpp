#include "learner.h"
#include <cmath>
#include <algorithm>

template<typename T>
static T clamp11(T val, T lo, T hi) {
    return val < lo ? lo : (val > hi ? hi : val);
}

ThresholdLearner::ThresholdLearner(double rssi_th, double int_th, double sim_th)
    : rssi_th_(rssi_th), int_th_(int_th), sim_th_(sim_th),
      rng_(std::random_device{}()),
      fp_rate_ema_(0.0), fn_rate_ema_(0.0),
      last_update_time_(0.0)
{}

void ThresholdLearner::addObservation(bool true_rogue, bool pred_anomaly,
                                       double rssi, double x, double y, double ts) {
    history_.push_back({true_rogue, pred_anomaly, rssi, x, y, ts});
    if (history_.size() > MAX_HISTORY) history_.pop_front();
    last_update_time_ = ts;
}

void ThresholdLearner::update() {
    if (history_.size() < 100) return;

    int fp = 0, fn = 0;
    for (const auto& fb : history_) {
        if (!fb.true_rogue &&  fb.pred_anomaly) ++fp;
        if ( fb.true_rogue && !fb.pred_anomaly) ++fn;
    }
    double n       = static_cast<double>(history_.size());
    double fp_rate = fp / n;
    double fn_rate = fn / n;

    const double alpha = 0.3;
    fp_rate_ema_ = (fp_rate_ema_ == 0.0) ? fp_rate : alpha * fp_rate + (1 - alpha) * fp_rate_ema_;
    fn_rate_ema_ = (fn_rate_ema_ == 0.0) ? fn_rate : alpha * fn_rate + (1 - alpha) * fn_rate_ema_;

    double fpe = fp_rate_ema_;
    double fne = fn_rate_ema_;

    // Desired targets (aggressive when FN high)
    double target_rssi = rssi_th_;
    double target_int  = int_th_;

    if (fne > 0.02) {
        target_rssi *= 0.98;    // more sensitive (lower)
        target_int  *= 1.03;    // much more sensitive on interval (raise)
    } else if (fne < 0.005) {
        target_rssi *= 1.005;
        target_int  *= 0.997;
    }

    if (fpe > 0.02) {
        target_rssi *= 1.01;    // less sensitive
        target_int  *= 0.98;
    } else if (fpe < 0.005) {
        target_rssi *= 0.997;
        target_int  *= 1.005;
    }

    // Absolute change limits (prevent whiplash)
    const double MAX_RSSI_CHANGE = 0.5;   // per update (5s)
    const double MAX_INT_CHANGE  = 0.1;

    double new_rssi = clamp11(target_rssi, rssi_th_ - MAX_RSSI_CHANGE, rssi_th_ + MAX_RSSI_CHANGE);
    double new_int  = clamp11(target_int,  int_th_  - MAX_INT_CHANGE,  int_th_  + MAX_INT_CHANGE);

    rssi_th_ = new_rssi;
    int_th_  = new_int;

    // Clamp to safe overall ranges (RSSI cap 6.5 enforced later in engine)
    rssi_th_ = clamp11(rssi_th_,  2.0, 20.0);   // engine will further cap
    int_th_  = clamp11(int_th_,  0.02, 2.0);
    sim_th_  = clamp11(sim_th_,  0.50, 0.95);
}

double ThresholdLearner::recentAnomalyRate() const {
    if (history_.empty()) return 0.0;
    std::size_t n = 0;
    for (const auto& fb : history_) if (fb.pred_anomaly) ++n;
    return static_cast<double>(n) / static_cast<double>(history_.size());
}

double ThresholdLearner::recentFPRate() const {
    if (history_.empty()) return 0.0;
    std::size_t n = 0;
    for (const auto& fb : history_) if (!fb.true_rogue && fb.pred_anomaly) ++n;
    return static_cast<double>(n) / static_cast<double>(history_.size());
}

double ThresholdLearner::recentFNRate() const {
    if (history_.empty()) return 0.0;
    std::size_t n = 0;
    for (const auto& fb : history_) if (fb.true_rogue && !fb.pred_anomaly) ++n;
    return static_cast<double>(n) / static_cast<double>(history_.size());
}
