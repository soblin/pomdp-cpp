#include <algorithm>
#include <iostream>
#include <numeric>

#include <xtensor/xio.hpp>

#include "pomdp-cpp/pomdp.h"

static constexpr int r = 60;
static constexpr int light_loc = 10;

using namespace pomdp_cpp;

class SimpleLightDark : public POMDPModel<int, int, double> {
public:
  SimpleLightDark(const States &states, const Actions &actions,
                  const Observations &observations, double discount = 0.95)
      : POMDPModel(states, actions, observations, discount) {}

  ptr<distribution::Distribution<State>>
  transition(const State &s, const Action &a) const override {
    return (a == 0) ? make_ptr<distribution::Deterministic<State>>(r + 1)
                    : make_ptr<distribution::Deterministic<State>>(
                          std::clamp(s + a, -r, r));
  }

  ptr<distribution::Distribution<Observation>>
  observation(const State &s, const Action &a, const State &sp) const override {
    return make_ptr<distribution::Normal<Observation>>(
        sp, std::fabs(sp - light_loc) + 0.0001);
  }

  double reward(const State &s, const Action &a) const override {
    if (a == 0)
      return s == 0 ? 100 : -100;
    else
      return -1.0;
  }

  bool is_terminal(const State &s) const override { return s < -r or s > r; }

  States sub_actions(const State &s) const { return actions(); }
};

int main() {
  std::vector<int> states(2 * r + 2);
  std::iota(states.begin(), states.end(),
            -r); // -r:r+1,  r+1 is a terminal state
  std::vector<int> actions({-10, -1, 0, 1, 10});
  auto simple_lightdark =
      SimpleLightDark(states, actions, std::vector<double>({}), 0.95);
  simple_lightdark.set_initial_state(
      distribution::Categorical<SimpleLightDark::State>({-r / 2, r / 2},
                                                        {0.5, 0.5})
          .draw());
  using Solver = ValueIterationSolver<int, int, double>;
  Solver solver(simple_lightdark);
  solver.solve();
  // std::cout << solver.qmat() << std::endl;
  return 0;
}
