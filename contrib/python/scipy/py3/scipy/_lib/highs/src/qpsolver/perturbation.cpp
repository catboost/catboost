#include "perturbation.hpp"

#include <random>

void perturb(Runtime& rt) {
  rt.perturbed = rt.instance;
  if (!rt.settings.perturbation) {
    return;
  }

  std::uniform_real_distribution<double> randomperturb(10E-6, 10E-5);
  std::default_random_engine re;

  for (HighsInt i = 0; i < rt.perturbed.num_con; i++) {
    if (rt.perturbed.con_lo[i] != rt.perturbed.con_up[i]) {
      if (rt.perturbed.con_lo[i] != -std::numeric_limits<double>::infinity()) {
        rt.perturbed.con_lo[i] -= randomperturb(re);
      }
      if (rt.perturbed.con_up[i] != std::numeric_limits<double>::infinity()) {
        rt.perturbed.con_up[i] += randomperturb(re);
      }
    }
  }
  for (HighsInt i = 0; i < rt.perturbed.num_var; i++) {
    if (rt.perturbed.var_lo[i] != rt.perturbed.var_up[i]) {
      if (rt.perturbed.var_lo[i] != -std::numeric_limits<double>::infinity()) {
        rt.perturbed.var_lo[i] -= randomperturb(re);
      }
      if (rt.perturbed.var_up[i] != std::numeric_limits<double>::infinity()) {
        rt.perturbed.var_up[i] += randomperturb(re);
      }
    }
  }
}