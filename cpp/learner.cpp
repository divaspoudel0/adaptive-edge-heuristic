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

    // Compute raw FP and FN counts over the whole history
    int fp = 0, fn = 0;
    for (const auto& fb : history_) {
        if (!fb.true_rogue &&  fb.pred_anomaly) ++fp;
        if ( fb.true_rogue && !fb.pred_anomaly) ++fn;
    }
    double n       = static_cast<double>(history_.size());
    double fp_rate = fp / n;
    double fn_rate = fn / n;

    // Update exponential moving averages (smoothing factor 0.3)
    const double alpha = 0.3;
    fp_rate_ema_ = (fp_rate_ema_ == 0.0) ? fp_rate : alpha * fp_rate + (1 - alpha) * fp_rate_ema_;
    fn_rate_ema_ = (fn_rate_ema_ == 0.0) ? fn_rate : alpha * fn_rate + (1 - alpha) * fn_rate_ema_;

    // Use the smoothed rates for decision
    double fpe = fp_rate_ema_;
    double fne = fn_rate_ema_;

    // Base adjustment factors (max relative change per update)
    const double MAX_INCREASE = 1.02;  // can increase at most 2% (loosen)
    const double MAX_DECREASE = 0.98;  // can decrease at most 2% (tighten)

    // Asymmetric: tightening (decrease) is easier than loosening (increase)
    // For FN-driven adjustments (need more sensitivity):
    //   - Lower rssi_th (more sensitive) → decrease factor, allowed full 2%
    //   - Raise int_th (more sensitive)  → increase factor, but limited to 1% because we want to be cautious
    // For FP-driven adjustments (need less sensitivity):
    //   - Raise rssi_th (less sensitive) → increase factor, limited to 1%
    //   - Lower int_th (less sensitive)  → decrease factor, allowed full 2%

    double target_rssi = rssi_th_;
    double target_int  = int_th_;

    // --- Handle high false negatives (missed rogues) ---
    if (fne > 0.02) {  // >2% FN
        // Make detection easier: lower rssi_th (tighten) and raise int_th (loosen interval sensitivity)
        target_rssi *= 0.98;               // decrease by up to 2%
        target_int  *= 1.01;                // increase by only 1% (cautious)
    }
    else if (fne < 0.005) {  // very low FN (<0.5%)
        // We can afford to be less sensitive: raise rssi_th slightly, lower int_th slightly
        target_rssi *= 1.005;               // increase by 0.5%
        target_int  *= 0.995;                // decrease by 0.5%
    }

    // --- Handle high false positives (false alarms) ---
    if (fpe > 0.02) {  // >2% FP
        // Make detection harder: raise rssi_th (loosen) and lower int_th (tighten interval)
        target_rssi *= 1.01;                // increase by only 1% (cautious)
        target_int  *= 0.98;                 // decrease by up to 2%
    }
    else if (fpe < 0.005) {  // very low FP (<0.5%)
        // We can be more sensitive: lower rssi_th, raise int_th
        target_rssi *= 0.995;                // decrease by 0.5%
        target_int  *= 1.005;                 // increase by 0.5%
    }

    // Apply slew rate limiting: ensure changes are within allowed bounds
    double rssi_change = target_rssi / rssi_th_;
    if (rssi_change > MAX_INCREASE) rssi_change = MAX_INCREASE;
    if (rssi_change < MAX_DECREASE) rssi_change = MAX_DECREASE;
    rssi_th_ *= rssi_change;

    double int_change = target_int / int_th_;
    if (int_change > MAX_INCREASE) int_change = MAX_INCREASE;
    if (int_change < MAX_DECREASE) int_change = MAX_DECREASE;
    int_th_ *= int_change;

    // Clamp to reasonable ranges
    rssi_th_ = clamp11(rssi_th_,  2.0, 20.0);
    int_th_  = clamp11(int_th_,  0.02,  1.5);
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
