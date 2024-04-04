#include "quass.hpp"

#include <algorithm>
#include <map>

#include "Highs.h"
#include "basis.hpp"
#include "crashsolution.hpp"
#include "dantzigpricing.hpp"
#include "devexharrispricing.hpp"
#include "devexpricing.hpp"
#include "factor.hpp"
#include "feasibility.hpp"
#include "gradient.hpp"
#include "instance.hpp"
#include "lp_data/HighsAnalysis.h"
#include "ratiotest.hpp"
#include "reducedcosts.hpp"
#include "reducedgradient.hpp"
#include "snippets.hpp"
#include "steepestedgepricing.hpp"
#include "scaling.hpp"
#include "perturbation.hpp"

void Quass::solve() {
  scale(runtime);
  runtime.instance = runtime.scaled;
  perturb(runtime);
  runtime.instance = runtime.perturbed;
  CrashSolution crash(runtime.instance.num_var, runtime.instance.num_con);
  computestartingpoint(runtime, crash);
  if (runtime.status != ProblemStatus::INDETERMINED) {
    return;
  }
  Basis basis(runtime, crash.active, crash.rowstatus, crash.inactive);
  solve(crash.primal, crash.rowact, basis);
}

Quass::Quass(Runtime& rt) : runtime(rt) {}

void Quass::loginformation(Runtime& rt, Basis& basis, CholeskyFactor& factor) {
  rt.statistics.iteration.push_back(rt.statistics.num_iterations);
  rt.statistics.nullspacedimension.push_back(rt.instance.num_var -
                                             basis.getnumactive());
  rt.statistics.objval.push_back(rt.instance.objval(rt.primal));
  rt.statistics.time.push_back(runtime.timer.readRunHighsClock());
  SumNum sm =
      rt.instance.sumnumprimalinfeasibilities(rt.primal, rt.rowactivity);
  rt.statistics.sum_primal_infeasibilities.push_back(sm.sum);
  rt.statistics.num_primal_infeasibilities.push_back(sm.num);
  rt.statistics.density_factor.push_back(factor.density());
  rt.statistics.density_nullspace.push_back(0.0);
}

static void tidyup(Vector& p, Vector& rowmove, Basis& basis, Runtime& runtime) {
  for (unsigned acon : basis.getactive()) {
    if (acon >= runtime.instance.num_con) {
      p.value[acon - runtime.instance.num_con] = 0.0;
    } else {
      rowmove.value[acon] = 0.0;
    }
  }
}

static void computerowmove(Runtime& runtime, Basis& basis, Vector& p,
                    Vector& rowmove) {
  runtime.instance.A.mat_vec(p, rowmove);
  return;
  // rowmove.reset();
  MatrixBase& Atran = runtime.instance.A.t();
  Atran.vec_mat(p, rowmove);
  return;
  for (HighsInt i = 0; i < runtime.instance.num_con; i++) {
    if (basis.getstatus(i) == BasisStatus::Default) {
      // check with assertions, is it really the same?
      double val =
          p.dot(&Atran.index[Atran.start[i]], &Atran.value[Atran.start[i]],
                Atran.start[i + 1] - Atran.start[i]);
      // Vector col = Atran.extractcol(i);
      // val = col * p;

      // assert(rowmove.value[i] == val);
      rowmove.value[i] = val;
    } else {
      rowmove.value[i] = 0;
    }
  }
  rowmove.resparsify();
}

// VECTOR
static Vector& computesearchdirection_minor(Runtime& rt, Basis& bas,
                                     CholeskyFactor& cf,
                                     ReducedGradient& redgrad, Vector& p) {
  Vector g2 = -redgrad.get();
  g2.sanitize();
  cf.solve(g2);

  g2.sanitize();

  return bas.Zprod(g2, p);
}

// VECTOR
static Vector& computesearchdirection_major(Runtime& runtime, Basis& basis,
                                     CholeskyFactor& factor, const Vector& yp,
                                     Gradient& gradient, Vector& gyp, Vector& l,
                                     Vector& m, Vector& p) {
  Vector yyp = yp;
  // if (gradient.getGradient().dot(yp) > 0.0) {
  //   yyp.scale(-1.0);
  // }
  runtime.instance.Q.mat_vec(yyp, gyp);
  if (basis.getnumactive() < runtime.instance.num_var) {
    basis.Ztprod(gyp, m);
    l = m;
    factor.solveL(l);
    Vector v = l;
    factor.solveLT(v);
    basis.Zprod(v, p);
    if (gradient.getGradient().dot(yyp) < 0.0) {
      return p.saxpy(-1.0, 1.0, yyp);
    } else {
      return p.saxpy(-1.0, -1.0, yyp);
    }

  } else {
    return p.repopulate(yp).scale(-gradient.getGradient().dot(yp));
    // return -yp;
  }
}

static double computemaxsteplength(Runtime& runtime, const Vector& p,
                            Gradient& gradient, Vector& buffer_Qp, bool& zcd) {
  double denominator = p * runtime.instance.Q.mat_vec(p, buffer_Qp);
  if (fabs(denominator) > 10E-5) {
    double numerator = -(p * gradient.getGradient());
    if (numerator < 0.0) {
      return 0.0;
    } else {
      return numerator / denominator;
    }
  } else {
    zcd = true;
    return std::numeric_limits<double>::infinity();
  }
}

static QpSolverStatus reduce(Runtime& rt, Basis& basis, const HighsInt newactivecon,
                      Vector& buffer_d, HighsInt& maxabsd,
                      HighsInt& constrainttodrop) {
  HighsInt idx = indexof(basis.getinactive(), newactivecon);
  if (idx != -1) {
    maxabsd = idx;
    constrainttodrop = newactivecon;
    Vector::unit(basis.getinactive().size(), idx, buffer_d);
    return QpSolverStatus::OK;
    // return NullspaceReductionResult(true);
  }

  // TODO: this operation is inefficient.
  Vector aq = rt.instance.A.t().extractcol(newactivecon);
  basis.Ztprod(aq, buffer_d, true, newactivecon);

  maxabsd = 0;
  for (HighsInt i = 0; i < buffer_d.num_nz; i++) {
    if (fabs(buffer_d.value[buffer_d.index[i]]) >
        fabs(buffer_d.value[maxabsd])) {
      maxabsd = buffer_d.index[i];
    }
  }
  constrainttodrop = basis.getinactive()[maxabsd];
  if (fabs(buffer_d.value[maxabsd]) < rt.settings.d_zero_threshold) {
    printf(
        "degeneracy? not possible to find non-active constraint to "
        "leave basis. max: log(d[%" HIGHSINT_FORMAT "]) = %lf\n",
        maxabsd, log10(fabs(buffer_d.value[maxabsd])));
    return QpSolverStatus::DEGENERATE;
  }
  return QpSolverStatus::OK;
  // return NullspaceReductionResult(idx != -1);
}

static std::unique_ptr<Pricing> getPricing(Runtime& runtime, Basis& basis,
                                    ReducedCosts& redcosts) {
  switch (runtime.settings.pricing) {
    case PricingStrategy::Devex:
      return std::unique_ptr<Pricing>(
          new DevexPricing(runtime, basis, redcosts));
    case PricingStrategy::DantzigWolfe:
      return std::unique_ptr<Pricing>(
          new DantzigPricing(runtime, basis, redcosts));
  }
  return nullptr;
}

static void regularize(Runtime& rt) {
  // add small diagonal to hessian
  for (HighsInt i = 0; i < rt.instance.num_var; i++) {
    for (HighsInt index = rt.instance.Q.mat.start[i];
         index < rt.instance.Q.mat.start[i + 1]; index++) {
      if (rt.instance.Q.mat.index[index] == i) {
        rt.instance.Q.mat.value[index] +=
            rt.settings.semidefiniteregularization;
      }
    }
  }
}

void Quass::solve(const Vector& x0, const Vector& ra, Basis& b0) {
  runtime.statistics.time_start = std::chrono::high_resolution_clock::now();
  Basis& basis = b0;
  runtime.primal = x0;

  // TODO: remove redundant equations before starting
  // HOWTO: from crash start, check all (near-)equality constraints (not
  // bounds). if the residual is 0 (or near-zero?), remove constraint

  Gradient gradient(runtime);
  ReducedCosts redcosts(runtime, basis, gradient);
  ReducedGradient redgrad(runtime, basis, gradient);
  CholeskyFactor factor(runtime, basis);
  runtime.instance.A.mat_vec(runtime.primal, runtime.rowactivity);
  std::unique_ptr<Pricing> pricing = getPricing(runtime, basis, redcosts);

  Vector p(runtime.instance.num_var);
  Vector rowmove(runtime.instance.num_con);

  Vector buffer_yp(runtime.instance.num_var);
  Vector buffer_gyp(runtime.instance.num_var);
  Vector buffer_l(runtime.instance.num_var);
  Vector buffer_m(runtime.instance.num_var);

  Vector buffer_Qp(runtime.instance.num_var);

  // buffers for reduction
  Vector buffer_d(runtime.instance.num_var);

  regularize(runtime);

  bool atfsep = basis.getnumactive() == runtime.instance.num_var;
  while (true) {
    // check iteration limit
    if (runtime.statistics.num_iterations >= runtime.settings.iterationlimit) {
      runtime.status = ProblemStatus::ITERATIONLIMIT;
      break;
    }

    // check time limit
    if (runtime.timer.readRunHighsClock() >= runtime.settings.timelimit) {
      runtime.status = ProblemStatus::TIMELIMIT;
      break;
    }

    // LOGGING
    if (runtime.statistics.num_iterations %
            runtime.settings.reportingfequency ==
        0) {
      loginformation(runtime, basis, factor);
      runtime.endofiterationevent.fire(runtime);
    }
    runtime.statistics.num_iterations++;

    QpSolverStatus status;

    bool zero_curvature_direction = false;
    double maxsteplength = 1.0;
    if (atfsep) {
      HighsInt minidx = pricing->price(runtime.primal, gradient.getGradient());
      if (minidx == -1) {
        runtime.status = ProblemStatus::OPTIMAL;
        break;
      }

      HighsInt unit = basis.getindexinfactor()[minidx];
      Vector::unit(runtime.instance.num_var, unit, buffer_yp);
      basis.btran(buffer_yp, buffer_yp, true, minidx);

      buffer_l.dim = basis.getnuminactive();
      buffer_m.dim = basis.getnuminactive();
      computesearchdirection_major(runtime, basis, factor, buffer_yp, gradient,
                                   buffer_gyp, buffer_l, buffer_m, p);
      basis.deactivate(minidx);
      computerowmove(runtime, basis, p, rowmove);
      tidyup(p, rowmove, basis, runtime);
      maxsteplength = std::numeric_limits<double>::infinity();
      // if (runtime.instance.Q.mat.value.size() > 0) {
      double denominator = p * runtime.instance.Q.mat_vec(p, buffer_Qp);
      maxsteplength = computemaxsteplength(runtime, p, gradient, buffer_Qp,
                                           zero_curvature_direction);
      if (!zero_curvature_direction) {
        status = factor.expand(buffer_yp, buffer_gyp, buffer_l, buffer_m);
        if (status != QpSolverStatus::OK) {
          runtime.status = ProblemStatus::INDETERMINED;
          return;
        }
      }
      redgrad.expand(buffer_yp);
    } else {
      computesearchdirection_minor(runtime, basis, factor, redgrad, p);
      computerowmove(runtime, basis, p, rowmove);
      tidyup(p, rowmove, basis, runtime);
    }

    if (p.norm2() < runtime.settings.pnorm_zero_threshold ||
        maxsteplength == 0.0) {
      atfsep = true;
    } else {
      RatiotestResult stepres = ratiotest(runtime, p, rowmove, maxsteplength);
      if (stepres.limitingconstraint != -1) {
        HighsInt constrainttodrop;
        HighsInt maxabsd;
        status = reduce(runtime, basis, stepres.limitingconstraint, buffer_d,
                        maxabsd, constrainttodrop);
        if (status != QpSolverStatus::OK) {
          runtime.status = ProblemStatus::INDETERMINED;
          return;
        }
        if (!zero_curvature_direction) {
          factor.reduce(
              buffer_d, maxabsd,
              indexof(basis.getinactive(), stepres.limitingconstraint) != -1);
        }
        redgrad.reduce(buffer_d, maxabsd);
        redgrad.update(stepres.alpha, false);

        status = basis.activate(runtime.settings, stepres.limitingconstraint,
                                stepres.nowactiveatlower
                                    ? BasisStatus::ActiveAtLower
                                    : BasisStatus::ActiveAtUpper,
                                constrainttodrop, pricing.get());
        if (status != QpSolverStatus::OK) {
          runtime.status = ProblemStatus::INDETERMINED;
          return;
        }
        if (basis.getnumactive() != runtime.instance.num_var) {
          atfsep = false;
        }
      } else {
        if (stepres.alpha ==
            std::numeric_limits<double>::infinity()) {
          // unbounded
          runtime.status = ProblemStatus::UNBOUNDED;
          return;
        }
        atfsep = false;
        redgrad.update(stepres.alpha, false);
      }

      gradient.update(buffer_Qp, stepres.alpha);
      redcosts.update();

      runtime.primal.saxpy(stepres.alpha, p);
      runtime.rowactivity.saxpy(stepres.alpha, rowmove);
    }
  }

  loginformation(runtime, basis, factor);
  runtime.endofiterationevent.fire(runtime);

  runtime.instance.sumnumprimalinfeasibilities(
      runtime.primal, runtime.instance.A.mat_vec(runtime.primal));

  Vector lambda = redcosts.getReducedCosts();
  for (auto e : basis.getactive()) {
    HighsInt indexinbasis = basis.getindexinfactor()[e];
    if (e >= runtime.instance.num_con) {
      // active variable bound
      HighsInt var = e - runtime.instance.num_con;
      runtime.dualvar.value[var] = lambda.value[indexinbasis];
    } else {
      runtime.dualcon.value[e] = lambda.value[indexinbasis];
    }
  }

  if (basis.getnumactive() == runtime.instance.num_var) {
    runtime.primal = basis.recomputex(runtime.instance);
  }
  // x.report("x");
  runtime.statistics.time_end = std::chrono::high_resolution_clock::now();
}
