#include "learner.h"
#include <cmath>
#include <cstddef>
#include <deque>

template<typename T> static T clamp11(T v, T lo, T hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

ThresholdLearner::ThresholdLearner(double r, double i, double s)
    : rssi_th_(r), int_th_(i), sim_th_(s), rng_(std::random_device{}()) {}

void ThresholdLearner::addObservation(bool tr, bool pa, double rssi,
                                       double x, double y, double ts) {
    history_.push_back({tr, pa, rssi, x, y, ts});
    if (history_.size() > MAX_HISTORY) history_.pop_front();
}

void ThresholdLearner::update() {
    if (history_.size() < 100) return;
    int fp = 0, fn = 0;
    for (const auto& f : history_) {
        if (!f.true_rogue &&  f.pred_anomaly) ++fp;
        if ( f.true_rogue && !f.pred_anomaly) ++fn;
    }
    double n = static_cast<double>(history_.size());
    double fp_rate = fp / n, fn_rate = fn / n;
    if      (fp_rate > 0.05) rssi_th_ *= 1.01;
    else if (fp_rate < 0.01) rssi_th_ *= 0.99;
    if      (fn_rate > 0.05) rssi_th_ *= 0.99;
    else if (fn_rate < 0.01) rssi_th_ *= 1.01;
    rssi_th_ = clamp11(rssi_th_,  2.0, 20.0);
    int_th_  = clamp11(int_th_,  0.05,  1.0);
    sim_th_  = clamp11(sim_th_,  0.50,  0.95);
}

double ThresholdLearner::recentAnomalyRate() const {
    if (history_.empty()) return 0.0;
    std::size_t n = 0;
    for (const auto& f : history_) if (f.pred_anomaly) ++n;
    return static_cast<double>(n) / static_cast<double>(history_.size());
}
double ThresholdLearner::recentFPRate() const {
    if (history_.empty()) return 0.0;
    std::size_t n = 0;
    for (const auto& f : history_) if (!f.true_rogue && f.pred_anomaly) ++n;
    return static_cast<double>(n) / static_cast<double>(history_.size());
}
double ThresholdLearner::recentFNRate() const {
    if (history_.empty()) return 0.0;
    std::size_t n = 0;
    for (const auto& f : history_) if (f.true_rogue && !f.pred_anomaly) ++n;
    return static_cast<double>(n) / static_cast<double>(history_.size());
}
