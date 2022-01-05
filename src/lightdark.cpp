#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>

#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>

#include "pomdp-cpp/pomdp.h"

static constexpr int r = 60;
static constexpr int light_loc = 10;

using namespace pomdp_cpp;

class SimpleLightDark : public POMDPModel<int, int, double> {
public:
  SimpleLightDark(std::span<State> states, std::span<Action> actions,
                  std::span<Observation> observations, double discount = 0.95)
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

void write_qmat(const xt::xtensor<double, 2> &qmat) {
  std::ofstream ofs("ans_cpp.csv");
  if (!ofs) {
    std::cout << "failed to open ans.csv" << std::endl;
    return;
  }
  auto shape = qmat.shape();
  for (unsigned i = 0; i < shape[0]; ++i) {
    for (unsigned j = 0; j < shape[1]; ++j) {
      ofs << qmat(i, j);
      if (j != shape[1] - 1)
        ofs << ", ";
    }
    ofs << '\n';
  }
  ofs.close();
}

int main() {
  std::vector<int> states(2 * r + 2);
  std::iota(states.begin(), states.end(),
            -r); // -r:r+1,  r+1 is a terminal state
  std::vector<int> actions({-10, -1, 0, 1, 10});
  std::vector<double> observations({});
  auto simple_lightdark = SimpleLightDark(std::span{states}, std::span{actions},
                                          std::span{observations}, 0.95);
  simple_lightdark.set_initial_state(
      distribution::Categorical<SimpleLightDark::State>({-r / 2, r / 2},
                                                        {0.5, 0.5})
          .draw());
  using Solver = ValueIterationSolver<int, int, double>;
  Solver solver(simple_lightdark);
  solver.solve();
  auto policy = AlphaVectorPolicy<int, int, double>(solver.qmat());
  // write_qmat(solver.qmat());
  return 0;
}
