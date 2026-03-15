#pragma once
#include <deque>
#include <random>

struct Feedback {
    bool true_rogue, pred_anomaly;
    double rssi, x, y, timestamp;
};

class ThresholdLearner {
public:
    ThresholdLearner(double rssi_th, double int_th, double sim_th);
    void addObservation(bool true_rogue, bool pred_anomaly,
                        double rssi, double x, double y, double ts);
    void update();
    double rssiTh() const { return rssi_th_; }
    double intTh()  const { return int_th_;  }
    double simTh()  const { return sim_th_;  }
    double recentAnomalyRate() const;
    double recentFPRate() const;
    double recentFNRate() const;
    void setThresholds(double r, double i, double s) {
        rssi_th_ = r; int_th_ = i; sim_th_ = s;
    }
private:
    double rssi_th_, int_th_, sim_th_;
    std::deque<Feedback> history_;
    static const std::size_t MAX_HISTORY = 2000;
    std::mt19937 rng_;
};
