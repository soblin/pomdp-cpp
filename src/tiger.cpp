#include <iostream>
#include <string_view>

#include <xtensor/xio.hpp>

#include "pomdp-cpp/pomdp.h"

using namespace pomdp_cpp;

class Tiger
    : public POMDPModel<std::string_view, std::string_view, std::string_view> {
public:
  Tiger(const std::initializer_list<State> &states,
        const std::initializer_list<Action> &actions,
        const std::initializer_list<Observation> &observations,
        double discount = 0.95)
      : POMDPModel(states, actions, observations, discount) {}

  ptr<distribution::Distribution<State>>
  transition(const State &s, const Action &a) const override {
    if (a == "listen")
      return make_ptr<distribution::Deterministic<State>>(s);
    else
      return make_ptr<distribution::Categorical<State>>(
          std::vector<State>({"left", "right"}),
          std::vector<double>({0.5, 0.5}));
  }

  ptr<distribution::Distribution<Observation>>
  observation(const State &s, const Action &a, const State &sp) const override {
    if (a == "listen") {
      if (s == "left")
        return make_ptr<distribution::Categorical<Observation>>(
            observations(), std::vector<double>({0.85, 0.15}));
      else
        return make_ptr<distribution::Categorical<Observation>>(
            observations(), std::vector<double>({0.15, 0.85}));
    } else
      return make_ptr<distribution::Categorical<Observation>>(
          observations(), std::vector<double>({0.5, 0.5}));
  }

  double reward(const State &s, const Action &a) const override {
    if (a == "listen")
      return -1.0;
    else if (s == a)
      return -100.0;
    else
      return 10.0;
  }

  bool is_terminal(const State &s) const override { return false; }

  States sub_actions(const State &s) const { return actions(); }
};

int main() {
  auto m = Tiger({"left", "right"}, {"left", "right", "listen"},
                 {"left", "right"}); // from initializer_list
  m.set_initial_state(
      distribution::Categorical<Tiger::State>({"left", "right"}, {0.5, 0.5})
          .draw());

  using Solver =
      ValueIterationSolver<std::string_view, std::string_view,
                           std::string_view>; // TODO: repeating these S,A,O is
                                              // cumbersome. like
                                              // ValueIterationSolver<Tiger>

  Solver solver(m);
  solver.solve();
  std::cout << solver.qmat() << std::endl;
  return 0;
}
