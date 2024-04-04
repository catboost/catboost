/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Written and engineered 2008-2022 at the University of Edinburgh    */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/*    Authors: Julian Hall, Ivet Galabova, Leona Gottwald and Michael    */
/*    Feldmeier                                                          */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "presolve/ICrashX.h"

#include <algorithm>
#include <iostream>

#include "HConfig.h"
#include "io/HighsIO.h"
#include "ipm/IpxWrapper.h"

HighsStatus callCrossover(const HighsOptions& options, const HighsLp& lp,
                          HighsBasis& highs_basis,
                          HighsSolution& highs_solution,
                          HighsModelStatus& model_status,
                          HighsInfo& highs_info) {
  ipx::Int num_col, num_row;
  std::vector<ipx::Int> Ap, Ai;
  std::vector<double> objective, col_lb, col_ub, Av, rhs;
  std::vector<char> constraint_type;

  fillInIpxData(lp, num_col, num_row, objective, col_lb, col_ub, Ap, Ai, Av,
                rhs, constraint_type);
  // if (res != IpxStatus::OK) return HighsStatus::kError;

  const HighsLogOptions& log_options = options.log_options;

  ipx::Parameters parameters;
  parameters.crossover = true;
  parameters.crash_basis = 1;  // 0 = slack basis; 1 = crash basis
  parameters.display = 1;
  if (!options.output_flag) parameters.display = 0;
  // Modify parameters.debug according to log_dev_level
  parameters.debug = 0;
  if (options.log_dev_level == kHighsLogDevLevelDetailed) {
    parameters.debug = 0;
  } else if (options.log_dev_level == kHighsLogDevLevelInfo) {
    parameters.debug = 2;
  } else if (options.log_dev_level == kHighsLogDevLevelVerbose) {
    parameters.debug = 4;
  }

  ipx::LpSolver lps;
  lps.SetParameters(parameters);

  ipx::Int load_status =
      lps.LoadModel(num_col, &objective[0], &col_lb[0], &col_ub[0], num_row,
                    &Ap[0], &Ai[0], &Av[0], &rhs[0], &constraint_type[0]);
  if (load_status != 0) {
    highsLogUser(log_options, HighsLogType::kError,
                 "Error loading ipx model\n");
    return HighsStatus::kError;
  }

  // Set x values within bounds.
  std::vector<double> x(highs_solution.col_value);
  for (int i = 0; i < num_col; i++) {
    x[i] = std::max(x[i], col_lb[i]);
    x[i] = std::min(x[i], col_ub[i]);
  }

  // Build slack variables from rhs-A*x but subject to sign conditions.
  std::vector<double> slack(rhs);
  for (int i = 0; i < num_col; i++) {
    for (int p = Ap[i]; p < Ap[i + 1]; ++p) slack[Ai[p]] -= Av[p] * x[i];
  }
  for (int i = 0; i < num_row; i++) {
    switch (constraint_type[i]) {
      case '=':
        slack[i] = 0.0;
        break;
      case '<':
        slack[i] = std::max(slack[i], 0.0);
        break;
      case '>':
        slack[i] = std::min(slack[i], 0.0);
        break;
    }
  }
  ipx::Int crossover_status;
  if (highs_solution.dual_valid && highs_solution.col_dual.size() == num_col &&
      highs_solution.row_dual.size() == num_row) {
    highsLogUser(log_options, HighsLogType::kInfo,
                 "Calling IPX crossover with primal and dual values\n");
    crossover_status = lps.CrossoverFromStartingPoint(
        &x[0], &slack[0], &highs_solution.row_dual[0],
        &highs_solution.col_dual[0]);
  } else {
    highsLogUser(log_options, HighsLogType::kInfo,
                 "Calling IPX crossover with only primal values\n");
    crossover_status =
        lps.CrossoverFromStartingPoint(&x[0], &slack[0], nullptr, nullptr);
  }

  if (crossover_status != 0) {
    highsLogUser(log_options, HighsLogType::kError,
                 "IPX crossover error: flag = %d\n", (int)crossover_status);
    return HighsStatus::kError;
  }
  ipx::Info ipx_info = lps.GetInfo();
  highs_info.crossover_iteration_count += (HighsInt)ipx_info.updates_crossover;
  const bool imprecise_solution =
      ipx_info.status_crossover == IPX_STATUS_imprecise;
  if (ipx_info.status_crossover != IPX_STATUS_optimal &&
      ipx_info.status_crossover != IPX_STATUS_imprecise &&
      ipx_info.status_crossover != IPX_STATUS_time_limit) {
    highsLogUser(log_options, HighsLogType::kError,
                 "IPX crossover failed: status = %d\n",
                 (int)ipx_info.status_crossover);
    return HighsStatus::kError;
  }
  if (ipx_info.status_crossover == IPX_STATUS_time_limit) {
    model_status = HighsModelStatus::kTimeLimit;
    return HighsStatus::kWarning;
  }

  // Get basis
  IpxSolution ipx_solution;
  ipx_solution.num_col = num_col;
  ipx_solution.num_row = num_row;
  ipx_solution.ipx_col_value.resize(num_col);
  ipx_solution.ipx_row_value.resize(num_row);
  ipx_solution.ipx_col_dual.resize(num_col);
  ipx_solution.ipx_row_dual.resize(num_row);
  ipx_solution.ipx_row_status.resize(num_row);
  ipx_solution.ipx_col_status.resize(num_col);
  ipx::Int errflag = lps.GetBasicSolution(
      &ipx_solution.ipx_col_value[0], &ipx_solution.ipx_row_value[0],
      &ipx_solution.ipx_row_dual[0], &ipx_solution.ipx_col_dual[0],
      &ipx_solution.ipx_row_status[0], &ipx_solution.ipx_col_status[0]);
  if (errflag != 0) {
    highsLogUser(log_options, HighsLogType::kError,
                 "IPX crossover getting basic solution: flag = %d\n",
                 (int)errflag);
    return HighsStatus::kError;
  }

  // Convert the IPX basic solution to a HiGHS basic solution
  HighsStatus status = ipxBasicSolutionToHighsBasicSolution(
      options.log_options, lp, rhs, constraint_type, ipx_solution, highs_basis,
      highs_solution);

  if (status != HighsStatus::kOk) {
    highsLogUser(
        log_options, HighsLogType::kError,
        "Failed to convert IPX basic solution to Highs basic solution\n");
    return HighsStatus::kError;
  }
  highs_info.basis_validity =
      highs_basis.valid ? kBasisValidityValid : kBasisValidityInvalid;
  HighsStatus return_status;
  if (imprecise_solution) {
    model_status = HighsModelStatus::kUnknown;
    return_status = HighsStatus::kWarning;
  } else {
    model_status = HighsModelStatus::kOptimal;
    return_status = HighsStatus::kOk;
  }
  return return_status;
}
