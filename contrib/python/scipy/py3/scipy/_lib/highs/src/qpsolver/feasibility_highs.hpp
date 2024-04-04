#ifndef __SRC_LIB_FEASIBILITYHIGHS_HPP__
#define __SRC_LIB_FEASIBILITYHIGHS_HPP__

#include "Highs.h"
#include "crashsolution.hpp"

static void computestartingpoint_highs(Runtime& runtime, CrashSolution& result) {
  // compute initial feasible point
  Highs highs;

  // set HiGHS to be silent
  highs.setOptionValue("output_flag", false);
  highs.setOptionValue("presolve", "on");
  highs.setOptionValue("time_limit", runtime.settings.timelimit -
                                         runtime.timer.readRunHighsClock());

  HighsLp lp;
  lp.a_matrix_.index_ =
      *((std::vector<HighsInt>*)&runtime.instance.A.mat.index);
  lp.a_matrix_.start_ =
      *((std::vector<HighsInt>*)&runtime.instance.A.mat.start);
  lp.a_matrix_.value_ = runtime.instance.A.mat.value;
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.col_cost_.assign(runtime.instance.num_var, 0.0);
  // lp.col_cost_ = runtime.instance.c.value;
  lp.col_lower_ = runtime.instance.var_lo;
  lp.col_upper_ = runtime.instance.var_up;
  lp.row_lower_ = runtime.instance.con_lo;
  lp.row_upper_ = runtime.instance.con_up;
  lp.num_col_ = runtime.instance.num_var;
  lp.num_row_ = runtime.instance.num_con;

  // create artificial bounds for free variables
  if (runtime.settings.phase1boundfreevars) {
    for (HighsInt i=0; i<runtime.instance.num_var; i++) {
      if (isfreevar(runtime, i)) {
        lp.col_lower_[i] = -1E5;
        lp.col_upper_[i] = 1E5;
      }
    }
  }

  highs.passModel(lp);

  if (runtime.settings.phase1movefreevarsbasic) {
    HighsBasis basis;
    basis.alien = true;  // Set true when basis is instantiated
    for (HighsInt i = 0; i < runtime.instance.num_con; i++) {
      basis.row_status.push_back(HighsBasisStatus::kNonbasic);
    }

    for (HighsInt i = 0; i < runtime.instance.num_var; i++) {
      // make free variables basic
      if (runtime.instance.var_lo[i] ==
              -std::numeric_limits<double>::infinity() &&
          runtime.instance.var_up[i] ==
              std::numeric_limits<double>::infinity()) {
        // free variable
        basis.col_status.push_back(HighsBasisStatus::kBasic);
      } else {
        basis.col_status.push_back(HighsBasisStatus::kNonbasic);
      }
    }

    highs.setBasis(basis);
    const HighsBasis& internal_basis = highs.getBasis();

    highs.setOptionValue("simplex_strategy", kSimplexStrategyPrimal);
  }

  HighsStatus status = highs.run();
  if (status != HighsStatus::kOk) {
    runtime.status = ProblemStatus::ERROR;
    return;
  }

  runtime.statistics.phase1_iterations = highs.getInfo().simplex_iteration_count;

  HighsModelStatus phase1stat = highs.getModelStatus();
  if (phase1stat == HighsModelStatus::kInfeasible) {
    runtime.status = ProblemStatus::INFEASIBLE;
    return;
  }

  HighsSolution sol = highs.getSolution();
  HighsBasis bas = highs.getBasis();

  Vector x0(runtime.instance.num_var);
  Vector ra(runtime.instance.num_con);
  for (HighsInt i = 0; i < x0.dim; i++) {
    if (fabs(sol.col_value[i]) > 10E-5) {
      x0.value[i] = sol.col_value[i];
      x0.index[x0.num_nz++] = i;
    }
  }

  for (HighsInt i = 0; i < ra.dim; i++) {
    if (fabs(sol.row_value[i]) > 10E-5) {
      ra.value[i] = sol.row_value[i];
      ra.index[ra.num_nz++] = i;
    }
  }

  std::vector<HighsInt> initialactive;
  std::vector<HighsInt> initialinactive;
  std::vector<BasisStatus> atlower;
  for (HighsInt i = 0; i < bas.row_status.size(); i++) {
    if (bas.row_status[i] == HighsBasisStatus::kLower) {
      initialactive.push_back(i);
      atlower.push_back(BasisStatus::ActiveAtLower);
    } else if (bas.row_status[i] == HighsBasisStatus::kUpper) {
      initialactive.push_back(i);
      atlower.push_back(BasisStatus::ActiveAtUpper);
    } else if (bas.row_status[i] != HighsBasisStatus::kBasic) {
      // printf("row %d nonbasic\n", i);
      initialinactive.push_back(runtime.instance.num_con + i);
    } else {
      assert(bas.row_status[i] == HighsBasisStatus::kBasic);
    }
  }

  for (HighsInt i = 0; i < bas.col_status.size(); i++) {
    if (bas.col_status[i] == HighsBasisStatus::kLower) {
      if (isfreevar(runtime, i)) {
        initialinactive.push_back(runtime.instance.num_con + i);
      } else {
        initialactive.push_back(i + runtime.instance.num_con);
        atlower.push_back(BasisStatus::ActiveAtLower);
      }
      
    } else if (bas.col_status[i] == HighsBasisStatus::kUpper) {
      if (isfreevar(runtime, i)) {
        initialinactive.push_back(runtime.instance.num_con + i);
      } else {
        initialactive.push_back(i + runtime.instance.num_con);
        atlower.push_back(BasisStatus::ActiveAtUpper);
      }
      
    } else if (bas.col_status[i] == HighsBasisStatus::kZero) {
      // printf("col %" HIGHSINT_FORMAT " free and set to 0 %" HIGHSINT_FORMAT
      // "\n", i, (HighsInt)bas.col_status[i]);
      initialinactive.push_back(runtime.instance.num_con + i);
    } else if (bas.col_status[i] != HighsBasisStatus::kBasic) {
      // printf("Column %" HIGHSINT_FORMAT " basis stus %" HIGHSINT_FORMAT "\n",
      // i, (HighsInt)bas.col_status[i]);
    } else {
      assert(bas.col_status[i] == HighsBasisStatus::kBasic);
    }
  }

  assert(initialactive.size() + initialinactive.size() ==
         runtime.instance.num_var);

  for (HighsInt ia : initialinactive) {
    if (ia < runtime.instance.num_con) {
      printf("free row %d\n", (int)ia);
      assert(runtime.instance.con_lo[ia] ==
             -std::numeric_limits<double>::infinity());
      assert(runtime.instance.con_up[ia] ==
             std::numeric_limits<double>::infinity());
    } else {
      // printf("free col %d\n", (int)ia);
      assert(runtime.instance.var_lo[ia - runtime.instance.num_con] ==
             -std::numeric_limits<double>::infinity());
      assert(runtime.instance.var_up[ia - runtime.instance.num_con] ==
             std::numeric_limits<double>::infinity());
    }
  }

  result.rowstatus = atlower;
  result.active = initialactive;
  result.inactive = initialinactive;
  result.primal = x0;
  result.rowact = ra;
}

#endif
