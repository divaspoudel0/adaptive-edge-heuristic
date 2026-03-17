#pragma once
#include <deque>
#include <random>

struct Feedback {
    bool   true_rogue;
    bool   pred_anomaly;
    double rssi;
    double x, y;
    double timestamp;
};

class ThresholdLearner {
public:
    ThresholdLearner(double rssi_th, double int_th, double sim_th);

    void addObservation(bool true_rogue, bool pred_anomaly,
                        double rssi, double x, double y, double ts);
    void update();  // adjust thresholds based on recent FP/FN history

    double rssiTh() const { return rssi_th_; }
    double intTh()  const { return int_th_;  }
    double simTh()  const { return sim_th_;  }
    double recentAnomalyRate() const;
    double recentFPRate() const;
    double recentFNRate() const;

    void setThresholds(double rssi_th, double int_th, double sim_th) {
        rssi_th_ = rssi_th;
        int_th_  = int_th;
        sim_th_  = sim_th;
    }

private:
    double rssi_th_, int_th_, sim_th_;
    std::deque<Feedback>  history_;
    static const std::size_t MAX_HISTORY = 2000;

    std::mt19937 rng_;
    double sampleBeta(int a, int b); // not used, kept for compatibility

    // New EMA fields
    double fp_rate_ema_, fn_rate_ema_;
    double last_update_time_;
};
