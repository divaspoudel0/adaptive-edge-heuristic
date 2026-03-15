#include "learner.h"
#include <cmath>

// FIX: std::clamp is C++17 – provide C++11 version
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
    if (history_.size() > MAX_HISTORY)
        history_.pop_front();
}

void ThresholdLearner::update() {
    if (history_.size() < 100) return;

    int fp = 0, fn = 0;
    for (const auto& fb : history_) {
        if (!fb.true_rogue &&  fb.pred_anomaly) ++fp;   // false positive
        if ( fb.true_rogue && !fb.pred_anomaly) ++fn;   // false negative
    }

    double n        = static_cast<double>(history_.size());
    double fp_rate  = fp / n;
    double fn_rate  = fn / n;

    // Too many false positives → raise rssi_th (harder to trigger anomaly)
    // Too many false negatives → lower rssi_th (easier to trigger anomaly)
    if      (fp_rate > 0.05) rssi_th_ *= 1.01;
    else if (fp_rate < 0.01) rssi_th_ *= 0.99;

    if      (fn_rate > 0.05) rssi_th_ *= 0.99;
    else if (fn_rate < 0.01) rssi_th_ *= 1.01;

    // FIX: use clamp11 instead of std::clamp
    rssi_th_ = clamp11(rssi_th_,  2.0, 20.0);
    int_th_  = clamp11(int_th_,  0.05,  1.0);
    sim_th_  = clamp11(sim_th_,  0.50,  0.95);
}

double ThresholdLearner::recentAnomalyRate() const {
    if (history_.empty()) return 0.0;
    std::size_t anomalies = 0;
    for (const auto& fb : history_)
        if (fb.pred_anomaly) ++anomalies;
    return static_cast<double>(anomalies) / static_cast<double>(history_.size());
}

double ThresholdLearner::sampleBeta(int a, int b) {
    // Beta(a,b) via ratio of two Gamma samples
    std::gamma_distribution<double> ga(static_cast<double>(a), 1.0);
    std::gamma_distribution<double> gb(static_cast<double>(b), 1.0);
    double x = ga(rng_);
    double y = gb(rng_);
    if (x + y == 0.0) return 0.5;
    return x / (x + y);
}
