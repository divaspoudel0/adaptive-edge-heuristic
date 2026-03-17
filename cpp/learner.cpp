#include "learner.h"
#include <cmath>
#include <cstddef>
#include <deque>

template<typename T>
static T clamp11(T val, T lo, T hi) {
    return val < lo ? lo : (val > hi ? hi : val);
}

ThresholdLearner::ThresholdLearner(double rssi_th, double int_th, double sim_th)
    : rssi_th_(rssi_th), int_th_(int_th), sim_th_(sim_th),
      rng_(std::random_device{}())
{}

void ThresholdLearner::addObservation(bool true_rogue, bool pred_anomaly,
                                       double rssi, double x, double y, double ts) {
    history_.push_back({true_rogue, pred_anomaly, rssi, x, y, ts});
    if (history_.size() > MAX_HISTORY) history_.pop_front();
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

    // High FP → make anomaly harder to trigger (raise rssi_th)
    if      (fp_rate > 0.05) rssi_th_ *= 1.01;
    else if (fp_rate < 0.01) rssi_th_ *= 0.99;

    // High FN → make anomaly easier to trigger (lower rssi_th)
    if      (fn_rate > 0.05) rssi_th_ *= 0.99;
    else if (fn_rate < 0.01) rssi_th_ *= 1.01;

    rssi_th_ = clamp11(rssi_th_,  2.0, 20.0);
    int_th_  = clamp11(int_th_,  0.05,  1.0);
    sim_th_  = clamp11(sim_th_,  0.50,  0.95);
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

double ThresholdLearner::sampleBeta(int a, int b) {
    std::gamma_distribution<double> ga(static_cast<double>(a), 1.0);
    std::gamma_distribution<double> gb(static_cast<double>(b), 1.0);
    double x = ga(rng_), y = gb(rng_);
    if (x + y == 0.0) return 0.5;
    return x / (x + y);
}
