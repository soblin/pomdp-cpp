#pragma once

#include <random>
#include <tuple>
#include <vector>

namespace pomdp_cpp::distribution {

template <typename T> class Distribution {
public:
  Distribution() = default;
  ~Distribution() = default;
  virtual T draw() const = 0;
  virtual const std::vector<std::tuple<T, double>> &sps() const = 0;
};

// TODO: pass std::random_device or seed engine as the argument ?
template <typename State> class Categorical : public Distribution<State> {
public:
  Categorical(const std::vector<State> &states,
              const std::vector<double> &probs)
      : states_(states), probs_(probs) {
    if (states.size() != probs.size()) {
      std::exit(0);
    }
    for (unsigned i = 0; i < states.size(); ++i) {
      sps_.emplace_back(states[i], probs[i]);
    }
  }
  ~Categorical() = default;
  State draw() const override {
    // https://stackoverflow.com/questions/6223355/static-variables-in-member-functions
    static std::mt19937 gen(std::random_device{}()); // this can be shared
    std::discrete_distribution<> d(probs_.begin(), probs_.end());
    int index = d(gen);
    return states_[index];
  }
  const std::vector<std::tuple<State, double>> &sps() const override {
    return sps_;
  }

private:
  const std::vector<State> states_;
  const std::vector<double> probs_;
  std::vector<std::tuple<State, double>> sps_;
};

template <typename State> class Deterministic : public Distribution<State> {
public:
  Deterministic(const State &state) : state_(state) {
    sps_.emplace_back(state, 1.0);
  }
  ~Deterministic() = default;
  State draw() const override { return state_; }
  const std::vector<std::tuple<State, double>> &sps() const override {
    return sps_;
  }

private:
  State state_;
  std::vector<std::tuple<State, double>> sps_;
};

// TODO: template <typename State, int dim> class MvNormal : public
// Distribution<State> {};
template <typename State> class Normal : public Distribution<State> {
public:
  Normal(const State &mu, const double sigma) : mu_(mu), sigma_(sigma) {
    sps_.emplace_back(mu_, 1.0);
  }
  ~Normal() = default;
  State draw() const override {
    static std::mt19937 gen(std::random_device{}());
    std::normal_distribution<> d(static_cast<double>(mu_), sigma_);
    return static_cast<State>(d(gen));
  }
  // TODO: should return belief particles
  const std::vector<std::tuple<State, double>> &sps() const override {
    return sps_;
  }

private:
  const State mu_;
  const double sigma_;
  std::vector<std::tuple<State, double>> sps_;
};

} // namespace pomdp_cpp::distribution
