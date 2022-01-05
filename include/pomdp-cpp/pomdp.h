#pragma once

#include <initializer_list>
#include <optional>
#include <span>
#include <type_traits>
#include <unordered_map>

#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include <pomdp-cpp/distribution.h>
#include <pomdp-cpp/util.h>

namespace pomdp_cpp {

template <typename S, typename A, typename O> class POMDPModel {
public:
  using State = S;
  using Action = A;
  using Observation = O;
  template <typename T>
  using Container = std::vector<T>; // should be vector or array
  using States = Container<S>;
  using Actions = Container<A>;
  using Observations = Container<O>;

  POMDPModel(std::span<State> states, std::span<Action> actions,
             std::span<Observation> observations, double discount = 0.95)
      : discount_(discount), initial_state_(std::nullopt) {
    for (const auto &state : states)
      states_.push_back(state);

    for (const auto &action : actions)
      actions_.push_back(action);

    for (const auto &observation : observations)
      observations_.push_back(observation);

    index_states_actions();
  }
  POMDPModel(const std::initializer_list<State> &states,
             const std::initializer_list<Action> &actions,
             const std::initializer_list<Observation> &observations,
             double discount = 0.95)
      : discount_(discount), initial_state_(std::nullopt) {
    for (const auto &state : states)
      states_.push_back(state);

    for (const auto &action : actions)
      actions_.push_back(action);

    for (const auto &observation : observations)
      observations_.push_back(observation);

    index_states_actions();
  }
  /// transition: in MDP, this should return a state. in POMDP, it should return
  /// a distribution(aka belief)
  virtual ptr<distribution::Distribution<State>>
  transition(const State &s, const Action &a) const = 0;

  /// observation: only in POMDP
  virtual ptr<distribution::Distribution<Observation>>
  observation(const State &s, const Action &a, const State &sp) const = 0;

  /// reward: TODO. it could take sp(next state) and observation as well.
  virtual double reward(const State &s, const Action &a) const = 0;

  /// sub_actions: possible next action from `s`
  virtual States sub_actions(const State &s) const = 0;

  virtual bool is_terminal(const State &s) const = 0;

  void set_initial_state(const State &s) { initial_state_ = s; }

  // interface for value iteration
  const States &states() const { return states_; }
  const Actions &actions() const { return actions_; }
  const Observations &observations() const { return observations_; }
  int state2ind(const State &state) { return state2ind_[state]; }
  int action2ind(const Action &action) { return action2ind_[action]; }
  double discount() { return discount_; }

private:
  States states_;
  Actions actions_;
  Observations observations_;
  double discount_;
  std::optional<State> initial_state_;
  std::unordered_map<State, int> state2ind_;
  std::unordered_map<Action, int> action2ind_;
  void index_states_actions() {
    int cnt = 0;
    for (auto &&state : states_)
      state2ind_[state] = cnt++;

    cnt = 0;
    for (auto &&action : actions_)
      action2ind_[action] = cnt++;
  }
};

template <typename S> class Belief {
public:
  Belief(std::span<double> probs_) {
    for (auto &&prob : probs_)
      probs.push_back(prob);
  }
  std::vector<double> probs;
  // private:
  // std::vector<int> istates; // istate of state that has non-zero prob
};

template <typename S, typename A, typename O> class AlphaVectorPolicy {
public:
  AlphaVectorPolicy(const xt::xtensor<double, 2> &qmat) : qmat_(qmat) {
    auto shape = qmat_.shape();
    ns_ = shape[0];
    na_ = shape[1];
  }
  double value(const Belief<S> &bel) {
    double max = -std::numeric_limits<double>::infinity();
    int max_a = -1;
    for (int i = 0; i < na_; ++i) {
      auto alpha = xt::col(qmat_, i);
      double val = 0;
      for (int j = 0; j < ns_; ++j)
        val += alpha(j) * bel.probs[j];
      if (val > max) {
        max = val;
        max_a = i;
      }
    }
    return max;
  }
  int iaction(const Belief<S> &bel) {
    double max = -std::numeric_limits<double>::infinity();
    int max_a = -1;
    for (int i = 0; i < na_; ++i) {
      auto alpha = xt::col(qmat_, i);
      double val = 0;
      for (int j = 0; j < ns_; ++j)
        val += alpha(j) * bel.probs[j];
      if (val > max) {
        max = val;
        max_a = i;
      }
    }
    return max_a;
  }
  // design
  // calc the internal product of each column of qmat and belief, and then the
  // corresponding action that gives the highest value. need to consider the
  // design of "belief" of state
private:
  const xt::xtensor<double, 2> &qmat_;
  int ns_;
  int na_;
};

template <typename S, typename A, typename O> class ValueIterationSolver {
public:
  using State = S;
  using Action = A;
  using Observation = O;
  ValueIterationSolver(POMDPModel<S, A, O> &model, int max_iterations = 100,
                       double bel_res = 0.001)
      : model_(model), max_iterations_(max_iterations), bel_res_(bel_res),
        discount_(model.discount()) {
    // https://xtensor.readthedocs.io/en/latest/quickref/basic.html#initialization
    auto ns = model_.states().size();
    auto na = model_.actions().size();
    xt::xtensor<double, 1>::shape_type shape_util = {ns};
    xt::xtensor<int, 1>::shape_type shape_pol = {ns};
    xt::xtensor<double, 2>::shape_type shape_qmat = {ns, na};
    util_ = xt::xtensor<double, 1>(shape_util);
    pol_ = xt::xtensor<int, 1>(shape_pol);
    qmat_ = xt::xtensor<double, 2>(shape_qmat);
  }

private:
  POMDPModel<S, A, O> &model_;
  const int max_iterations_;
  const double bel_res_;
  const double discount_;
  xt::xtensor<double, 1> util_; // istate => utility value
  xt::xtensor<int, 1> pol_;     // istate => iaction of opt-policy
  xt::xtensor<double, 2> qmat_; // [istate, iaction] => q-value

public:
  const xt::xtensor<double, 2> &qmat() { return qmat_; }
  void solve() {
    for (auto i = 0; i < max_iterations_; ++i) {
      double residual = 0.0;
      for (const auto &state : model_.states()) {
        auto istate = model_.state2ind(state);
        if (model_.is_terminal(state)) {
          util_[istate] = 0.0;
          // for convenience use 0
          // actually at terminal, "do nothing" (no action) is optimal
          pol_[istate] = 0;
          continue;
        }
        auto sub_actions = model_.sub_actions(state);
        // find optimal action
        double old_util = util_[istate];
        double max_util = -std::numeric_limits<double>::infinity();
        for (const auto &action : sub_actions) {
          auto iaction = model_.action2ind(action);
          auto dist = model_.transition(
              state, action); // create distribution over neighbors
          double u = 0.0;
          for (auto [s, p] : dist->sps()) {
            if (p == 0.0)
              continue;
            auto r = model_.reward(state, action);
            auto is = model_.state2ind(s);
            u += p * (r + discount_ * util_[is]);
          }
          double new_util = u;
          if (new_util > max_util) {
            max_util = new_util;
            pol_[istate] = iaction;
          }
          qmat_(istate, iaction) = new_util;
        }
        util_[istate] = max_util;
        double diff = std::fabs(max_util - old_util);
        if (diff > residual)
          residual = diff;
      }
    }
  }
};

} // namespace pomdp_cpp
