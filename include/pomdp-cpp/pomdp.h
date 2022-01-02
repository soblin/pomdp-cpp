#include <optional>
#include <type_traits>
#include <unordered_map>

#include <xtensor/xtensor.hpp>

#include "distribution.h"
#include "util.h"

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

  POMDPModel(const States &states, const Actions &actions,
             const Observations &observations, double discount = 0.95)
      : states_(states), actions_(actions), observations_(observations),
        discount_(discount), initial_state_(std::nullopt) {
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
  const States states_;
  const Actions actions_;
  const Observations observations_;
  const double discount_;
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

template <typename S, typename A, typename O> class AlphaVectorPolicy {};

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

  // pass qmat to AlphaVectorPolicy
  // template <template <class, class, class>
  //           class Policy> // like AlphaVectorPolciy
  // Policy<S, A, O> solve(const POMDPModel<S, A, O> &model);

private:
  POMDPModel<S, A, O> &model_;
  const int max_iterations_;
  const double bel_res_;
  const double discount_;
  xt::xtensor<double, 1> util_; // istate => utility value
  xt::xtensor<int, 1> pol_;     // istate => optimal policy
  xt::xtensor<double, 2> qmat_; // [istate, iaction] => q-value

public:
  const xt::xtensor<double, 2> &qmat() { return qmat_; }
  void solve() {
    for (auto i = 0; i < max_iterations_; ++i) {
      double residual = 0.0;
      for (const auto &state : model_.states()) {
        auto sub_actions = model_.sub_actions(state);
        auto istate = model_.state2ind(state);
        // find optimal aaction
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
