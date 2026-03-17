#include "learner.h"
#include <cmath>
#include <algorithm>

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

    // --- Adjust rssi_th ---
    // High FP → raise rssi_th (make anomaly harder)
    // High FN → lower rssi_th (make anomaly easier)
    if      (fp_rate > 0.02) rssi_th_ *= 1.02;  // more aggressive than before
    else if (fp_rate < 0.005) rssi_th_ *= 0.99;
    if      (fn_rate > 0.02) rssi_th_ *= 0.98;  // stronger reduction when many misses
    else if (fn_rate < 0.005) rssi_th_ *= 1.01;

    // --- Adjust int_th ---
    // High FP → lower int_th (make anomaly harder via interval)
    // High FN → raise int_th (make anomaly easier via interval)
    if      (fp_rate > 0.02) int_th_ *= 0.98;
    else if (fp_rate < 0.005) int_th_ *= 1.01;
    if      (fn_rate > 0.02) int_th_ *= 1.02;  // raise to catch erratic rogues
    else if (fn_rate < 0.005) int_th_ *= 0.99;

    // Clamp to reasonable ranges
    rssi_th_ = clamp11(rssi_th_,  2.0, 20.0);
    int_th_  = clamp11(int_th_,  0.02,  1.5);  // wider range for sensitivity
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
