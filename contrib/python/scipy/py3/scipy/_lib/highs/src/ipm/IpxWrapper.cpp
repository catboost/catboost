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
/**@file ipm/IpxWrapper.cpp
 * @brief
 * @author Julian Hall, Ivet Galabova, Qi Huangfu and Michael Feldmeier
 */
#include "ipm/IpxWrapper.h"

#include <cassert>

#include "lp_data/HighsOptions.h"
#include "lp_data/HighsSolution.h"

using std::min;

HighsStatus solveLpIpx(HighsLpSolverObject& solver_object) {
  return solveLpIpx(solver_object.options_, solver_object.timer_, solver_object.lp_, 
                    solver_object.basis_, solver_object.solution_, 
                    solver_object.model_status_, solver_object.highs_info_);
}

HighsStatus solveLpIpx(const HighsOptions& options,
		       HighsTimer& timer,
                       const HighsLp& lp, 
                       HighsBasis& highs_basis,
		       HighsSolution& highs_solution,
                       HighsModelStatus& model_status,
                       HighsInfo& highs_info) {
  // Use IPX to try to solve the LP
  //
  // Can return HighsModelStatus (HighsStatus) values:
  //
  // 1. kSolveError (kError) if various unlikely solution errors occur
  //
  // 2. kTimeLimit (kWarning) if time limit is reached
  //
  // 3. kIterationLimit (kWarning) if iteration limit is reached
  //
  // 4. kUnknown (kWarning) if IPM makes no progress or if
  // IPM/crossover are imprecise
  //
  // 5. kInfeasible (kOk) if IPM identifies primal infeasibility
  //
  // 6. kUnboundedOrInfeasible (kOk) if IPM identifies dual
  // infeasibility
  //
  // kOptimal (kOk) if IPM/crossover identify optimality
  //
  // With a non-error return, if just IPM has been run then a
  // non-vertex primal solution is obtained; if crossover has been run
  // then a basis and primal+dual solution are obtained.
  //
  //
  // Indicate that there is no valid primal solution, dual solution or basis
  highs_basis.valid = false;
  highs_solution.value_valid = false;
  highs_solution.dual_valid = false;
  // Indicate that no imprecise solution has (yet) been found
  resetModelStatusAndHighsInfo(model_status, highs_info);
  // Create the LpSolver instance
  ipx::LpSolver lps;
  // Set IPX parameters
  //
  // Cannot set internal IPX parameters directly since they are
  // private, so create instance of parameters
  ipx::Parameters parameters;
  // Set IPX parameters from options
  //
  // Set display according to output
  parameters.display = 1;
  if (!options.output_flag) parameters.display = 0;
  // Modify parameters.debug according to log_dev_level
  parameters.debug = 0;
  if (options.log_dev_level == kHighsLogDevLevelDetailed) {
    // Default options.log_dev_level setting is kHighsLogDevLevelNone, yielding
    // default setting debug = 0
    parameters.debug = 0;
  } else if (options.log_dev_level == kHighsLogDevLevelInfo) {
    parameters.debug = 2;
  } else if (options.log_dev_level == kHighsLogDevLevelVerbose) {
    parameters.debug = 4;
  }
  // Just test feasibility and optimality tolerances for now
  // ToDo Set more parameters
  parameters.ipm_feasibility_tol = min(options.primal_feasibility_tolerance,
                                       options.dual_feasibility_tolerance);

  parameters.ipm_optimality_tol = options.ipm_optimality_tolerance;
  parameters.crossover_start = options.start_crossover_tolerance;
  parameters.analyse_basis_data = kHighsAnalysisLevelNlaData & options.highs_analysis_level;
  // Determine the run time allowed for IPX
  parameters.time_limit = options.time_limit - timer.readRunHighsClock();
  parameters.ipm_maxiter = options.ipm_iteration_limit - highs_info.ipm_iteration_count;
  // Determine if crossover is to be run or not
  parameters.crossover = options.run_crossover;
  if (!parameters.crossover) {
    // If crossover is not run, then set crossover_start to -1 so that
    // IPX can terminate according to its feasibility and optimality
    // tolerances
    parameters.crossover_start = -1;
  }

  // Set the internal IPX parameters
  lps.SetParameters(parameters);

  ipx::Int num_col, num_row;
  std::vector<ipx::Int> Ap, Ai;
  std::vector<double> objective, col_lb, col_ub, Av, rhs;
  std::vector<char> constraint_type;
  fillInIpxData(lp, num_col, num_row, objective, col_lb, col_ub, Ap, Ai, Av,
                rhs, constraint_type);
  highsLogUser(options.log_options, HighsLogType::kInfo,
               "IPX model has %" HIGHSINT_FORMAT " rows, %" HIGHSINT_FORMAT
               " columns and %" HIGHSINT_FORMAT " nonzeros\n",
               num_row, num_col, Ap[num_col]);

  ipx::Int load_status =
      lps.LoadModel(num_col, &objective[0], &col_lb[0], &col_ub[0], num_row,
                    &Ap[0], &Ai[0], &Av[0], &rhs[0], &constraint_type[0]);

  if (load_status) {
    model_status = HighsModelStatus::kSolveError;
    return HighsStatus::kError;
  }

  // Use IPX to solve the LP!
  ipx::Int solve_status = lps.Solve();

  const bool report_solve_data = kHighsAnalysisLevelSolverSummaryData & options.highs_analysis_level;
  // Get solver and solution information.
  // Struct ipx_info defined in ipx/include/ipx_info.h
  const ipx::Info ipx_info = lps.GetInfo();
  if (report_solve_data) reportSolveData(options.log_options, ipx_info);
  highs_info.ipm_iteration_count += (HighsInt)ipx_info.iter;
  highs_info.crossover_iteration_count += (HighsInt)ipx_info.updates_crossover;

  // If not solved...
  if (solve_status != IPX_STATUS_solved) {
    const HighsStatus solve_return_status =
        reportIpxSolveStatus(options, solve_status, ipx_info.errflag);
    // Return error if IPX solve error has occurred
    if (solve_return_status == HighsStatus::kError) {
      model_status = HighsModelStatus::kSolveError;
      return HighsStatus::kError;
    }
  }
  bool ipm_status = true;
  const HighsStatus ipm_return_status =
      reportIpxIpmCrossoverStatus(options, ipx_info.status_ipm, ipm_status);
  ipm_status = false;
  const HighsStatus crossover_return_status = reportIpxIpmCrossoverStatus(
      options, ipx_info.status_crossover, ipm_status);
  // Return error if IPX IPM or crossover error has occurred
  if (ipm_return_status == HighsStatus::kError ||
      crossover_return_status == HighsStatus::kError) {
    model_status = HighsModelStatus::kSolveError;
    return HighsStatus::kError;
  }
  // Should only reach here if Solve() returned IPX_STATUS_solved or
  // IPX_STATUS_stopped
  if (ipxStatusError(solve_status != IPX_STATUS_solved &&
		     solve_status != IPX_STATUS_stopped,
		     options, "solve_status should be solved or stopped here but value is",
		     (int)solve_status))
    return HighsStatus::kError;

  // Only error returns so far
  //

  if (solve_status == IPX_STATUS_stopped) {
    // IPX stopped, so there's certainly no basic solution. Get the
    // non-vertex solution, though. This needs the model status to
    // know whether to worry about dual infeasibilities.
    const HighsModelStatus local_model_status = HighsModelStatus::kUnknown;
    getHighsNonVertexSolution(options, lp, num_col, num_row, rhs,
                              constraint_type, lps,
			      local_model_status, highs_solution);
    //
    // Look at the reason why IPX stopped
    //
    // Return error if stopped status settings occur that JAJH doesn't
    // think should happen
    //
    //==============
    // For crossover
    //==============
    if (illegalIpxStoppedCrossoverStatus(ipx_info, options))
      return HighsStatus::kError;
    // Can stop and reach time limit
    if (ipx_info.status_crossover == IPX_STATUS_time_limit) {
      model_status = HighsModelStatus::kTimeLimit;
      return HighsStatus::kWarning;
    }
    //========
    // For IPM
    //========
    //
    // Note that IPX can stop with IPM optimal, imprecise,
    // primal_infeas or dual_infeas, due to crossover stopping with
    // time limit, and this is why crossover returns are tested first
    if (illegalIpxStoppedIpmStatus(ipx_info, options))
      return HighsStatus::kError;
    // Can stop with time limit
    // Can stop with iter limit
    // Can stop with no progress
    if (ipx_info.status_ipm == IPX_STATUS_time_limit) {
      model_status = HighsModelStatus::kTimeLimit;
      return HighsStatus::kWarning;
    } else if (ipx_info.status_ipm == IPX_STATUS_iter_limit) {
      model_status = HighsModelStatus::kIterationLimit;
      return HighsStatus::kWarning;
    } else {
      assert(ipx_info.status_ipm == IPX_STATUS_no_progress);
      reportIpmNoProgress(options, ipx_info);
      model_status = HighsModelStatus::kUnknown;
      return HighsStatus::kWarning;
    }
  }
  // Should only reach here if Solve() returned IPX_STATUS_solved
  if (ipxStatusError(solve_status != IPX_STATUS_solved, options,
                     "solve_status should be solved here but value is",
                     (int)solve_status))
    return HighsStatus::kError;
  // Return error if solved status settings occur that JAJH doesn't
  // think should happen
  if (illegalIpxSolvedStatus(ipx_info, options)) return HighsStatus::kError;
  //==============
  // For crossover
  //==============
  // Can be not run
  // Can solve and be optimal
  // Can solve and be imprecise
  //========
  // For IPM
  //========
  // Can solve and be optimal
  // Can solve and be imprecise
  // Can solve and be primal_infeas
  // Can solve and be dual_infeas
  if (ipx_info.status_ipm == IPX_STATUS_primal_infeas ||
      ipx_info.status_ipm == IPX_STATUS_dual_infeas) {
    // IPM identified primal or dual infeasibility: crossover will not
    // have run, so get the non-vertex solution and return
    if (ipx_info.status_ipm == IPX_STATUS_primal_infeas) {
      model_status = HighsModelStatus::kInfeasible;
    } else if (ipx_info.status_ipm == IPX_STATUS_dual_infeas) {
      model_status = HighsModelStatus::kUnboundedOrInfeasible;
    }
    getHighsNonVertexSolution(options, lp, num_col, num_row, rhs,
                              constraint_type, lps,
			      model_status, highs_solution);
    return HighsStatus::kOk;
  }

  // Should only reach here if IPM is optimal or imprecise
  if (ipxStatusError(ipx_info.status_ipm != IPX_STATUS_optimal &&
                         ipx_info.status_ipm != IPX_STATUS_imprecise,
                     options,
                     "ipm status should be not run, optimal or imprecise "
                     "but value is",
                     (int)ipx_info.status_ipm))
    return HighsStatus::kError;

  // Should only reach here if crossover is not run, optimal or imprecise
  if (ipxStatusError(ipx_info.status_crossover != IPX_STATUS_not_run &&
		     ipx_info.status_crossover != IPX_STATUS_optimal &&
		     ipx_info.status_crossover != IPX_STATUS_imprecise,
                     options,
                     "crossover status should be not run, optimal or imprecise "
                     "but value is",
                     (int)ipx_info.status_crossover))
    return HighsStatus::kError;

  // Basic solution depends on crossover being run
  const bool have_basic_solution =
      ipx_info.status_crossover != IPX_STATUS_not_run;
  // Both crossover and IPM can be imprecise
  const bool imprecise_solution =
    ipx_info.status_crossover == IPX_STATUS_imprecise ||
    ipx_info.status_ipm == IPX_STATUS_imprecise;
  if (have_basic_solution) {
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
      highsLogUser(options.log_options, HighsLogType::kError,
		   "IPX crossover getting basic solution: flag = %d\n",
		   (int)errflag);
      return HighsStatus::kError;
    }
    // Convert the IPX basic solution to a HiGHS basic solution
    HighsStatus status = ipxBasicSolutionToHighsBasicSolution(options.log_options, lp, rhs,
                                         constraint_type, ipx_solution,
                                         highs_basis, highs_solution);
    if (status != HighsStatus::kOk) {
      highsLogUser(options.log_options, HighsLogType::kError,
		   "Failed to convert IPX basic solution to Highs basic solution\n");
      return HighsStatus::kError;
    }

  } else {
    // No basic solution, so get a non-vertex HiGHS solution. This
    // needs the model status to know whether to worry about dual
    // infeasibilities.
    const HighsModelStatus local_model_status = imprecise_solution ? HighsModelStatus::kUnknown : HighsModelStatus::kOptimal;
    getHighsNonVertexSolution(options, lp, num_col, num_row, rhs,
                              constraint_type, lps,
			      local_model_status, highs_solution);
    assert(!highs_basis.valid);
  }
  highs_info.basis_validity = highs_basis.valid ? kBasisValidityValid : kBasisValidityInvalid;
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

void fillInIpxData(const HighsLp& lp, ipx::Int& num_col, ipx::Int& num_row,
                   std::vector<double>& obj, std::vector<double>& col_lb,
                   std::vector<double>& col_ub, std::vector<ipx::Int>& Ap,
                   std::vector<ipx::Int>& Ai, std::vector<double>& Ax,
                   std::vector<double>& rhs,
                   std::vector<char>& constraint_type) {
  num_col = lp.num_col_;
  num_row = lp.num_row_;

  // For each row with both a lower and an upper bound introduce one new column
  // so num_col may increase. Ignore each free row so num_row may decrease.
  // lba <= a'x <= uba becomes
  // a'x-s = 0 and lba <= s <= uba.

  // For each row with bounds on both sides introduce explicit slack and
  // transfer bounds.
  assert((HighsInt)lp.row_lower_.size() == num_row);
  assert((HighsInt)lp.row_upper_.size() == num_row);

  std::vector<HighsInt> general_bounded_rows;
  std::vector<HighsInt> free_rows;

  for (HighsInt row = 0; row < num_row; row++)
    if (lp.row_lower_[row] < lp.row_upper_[row] &&
        lp.row_lower_[row] > -kHighsInf && lp.row_upper_[row] < kHighsInf)
      general_bounded_rows.push_back(row);
    else if (lp.row_lower_[row] <= -kHighsInf && lp.row_upper_[row] >= kHighsInf)
      free_rows.push_back(row);

  const HighsInt num_slack = general_bounded_rows.size();

  // For each row except free rows add entry to char array and set up rhs
  // vector
  rhs.reserve(num_row);
  constraint_type.reserve(num_row);

  for (int row = 0; row < num_row; row++) {
    if (lp.row_lower_[row] > -kHighsInf && lp.row_upper_[row] >= kHighsInf) {
      rhs.push_back(lp.row_lower_[row]);
      constraint_type.push_back('>');
    } else if (lp.row_lower_[row] <= -kHighsInf &&
               lp.row_upper_[row] < kHighsInf) {
      rhs.push_back(lp.row_upper_[row]);
      constraint_type.push_back('<');
    } else if (lp.row_lower_[row] == lp.row_upper_[row]) {
      rhs.push_back(lp.row_upper_[row]);
      constraint_type.push_back('=');
    } else if (lp.row_lower_[row] > -kHighsInf &&
               lp.row_upper_[row] < kHighsInf) {
      // general bounded
      rhs.push_back(0);
      constraint_type.push_back('=');
    }
  }

  std::vector<HighsInt> reduced_rowmap(lp.num_row_, -1);
  if (free_rows.size() > 0) {
    HighsInt counter = 0;
    HighsInt findex = 0;
    for (HighsInt row = 0; row < lp.num_row_; row++) {
      if (free_rows[findex] == row) {
        findex++;
        continue;
      } else {
        reduced_rowmap[row] = counter;
        counter++;
      }
    }
  } else {
    for (HighsInt k = 0; k < lp.num_row_; k++) reduced_rowmap[k] = k;
  }
  num_row -= free_rows.size();
  num_col += num_slack;

  std::vector<HighsInt> sizes(num_col, 0);

  for (HighsInt col = 0; col < lp.num_col_; col++)
    for (HighsInt k = lp.a_matrix_.start_[col]; k < lp.a_matrix_.start_[col + 1]; k++) {
      HighsInt row = lp.a_matrix_.index_[k];
      if (lp.row_lower_[row] > -kHighsInf || lp.row_upper_[row] < kHighsInf)
        sizes[col]++;
    }
  // Copy Astart and Aindex to ipx::Int array.
  HighsInt nnz = lp.a_matrix_.index_.size();
  Ap.resize(num_col + 1);
  Ai.reserve(nnz + num_slack);
  Ax.reserve(nnz + num_slack);

  // Set starting points of original and newly introduced columns.
  Ap[0] = 0;
  for (HighsInt col = 0; col < lp.num_col_; col++) {
    Ap[col + 1] = Ap[col] + sizes[col];
  }
  for (HighsInt col = lp.num_col_; col < (HighsInt)num_col; col++) {
    Ap[col + 1] = Ap[col] + 1;
  }
  for (HighsInt k = 0; k < nnz; k++) {
    HighsInt row = lp.a_matrix_.index_[k];
    if (lp.row_lower_[row] > -kHighsInf || lp.row_upper_[row] < kHighsInf) {
      Ai.push_back(reduced_rowmap[lp.a_matrix_.index_[k]]);
      Ax.push_back(lp.a_matrix_.value_[k]);
    }
  }

  for (HighsInt k = 0; k < num_slack; k++) {
    Ai.push_back((ipx::Int)general_bounded_rows[k]);
    Ax.push_back(-1);
  }

  // Column bound vectors.
  col_lb.resize(num_col);
  col_ub.resize(num_col);
  for (HighsInt col = 0; col < lp.num_col_; col++) {
    if (lp.col_lower_[col] <= -kHighsInf)
      col_lb[col] = -INFINITY;
    else
      col_lb[col] = lp.col_lower_[col];

    if (lp.col_upper_[col] >= kHighsInf)
      col_ub[col] = INFINITY;
    else
      col_ub[col] = lp.col_upper_[col];
  }
  for (HighsInt slack = 0; slack < num_slack; slack++) {
    const int row = general_bounded_rows[slack];
    col_lb[lp.num_col_ + slack] = lp.row_lower_[row];
    col_ub[lp.num_col_ + slack] = lp.row_upper_[row];
  }

  obj.resize(num_col);
  for (HighsInt col = 0; col < lp.num_col_; col++) {
    obj[col] = (HighsInt)lp.sense_ * lp.col_cost_[col];
  }
  obj.insert(obj.end(), num_slack, 0);
}

HighsStatus reportIpxSolveStatus(const HighsOptions& options,
                                 const ipx::Int solve_status,
                                 const ipx::Int error_flag) {
  if (solve_status == IPX_STATUS_solved) {
    highsLogUser(options.log_options, HighsLogType::kInfo, "Ipx: Solved\n");
    return HighsStatus::kOk;
  } else if (solve_status == IPX_STATUS_stopped) {
    highsLogUser(options.log_options, HighsLogType::kWarning, "Ipx: Stopped\n");
    return HighsStatus::kWarning;
  } else if (solve_status == IPX_STATUS_no_model) {
    if (error_flag == IPX_ERROR_argument_null) {
      highsLogUser(options.log_options, HighsLogType::kError,
                   "Ipx: Invalid input - argument_null\n");
      return HighsStatus::kError;
    } else if (error_flag == IPX_ERROR_invalid_dimension) {
      highsLogUser(options.log_options, HighsLogType::kError,
                   "Ipx: Invalid input - invalid dimension\n");
      return HighsStatus::kError;
    } else if (error_flag == IPX_ERROR_invalid_matrix) {
      highsLogUser(options.log_options, HighsLogType::kError,
                   "Ipx: Invalid input - invalid matrix\n");
      return HighsStatus::kError;
    } else if (error_flag == IPX_ERROR_invalid_vector) {
      highsLogUser(options.log_options, HighsLogType::kError,
                   "Ipx: Invalid input - invalid vector\n");
      return HighsStatus::kError;
    } else if (error_flag == IPX_ERROR_invalid_basis) {
      highsLogUser(options.log_options, HighsLogType::kError,
                   "Ipx: Invalid input - invalid basis\n");
      return HighsStatus::kError;
    } else {
      highsLogUser(options.log_options, HighsLogType::kError,
                   "Ipx: Invalid input - unrecognised error\n");
      return HighsStatus::kError;
    }
  } else if (solve_status == IPX_STATUS_out_of_memory) {
    highsLogUser(options.log_options, HighsLogType::kError,
                 "Ipx: Out of memory\n");
    return HighsStatus::kError;
  } else if (solve_status == IPX_STATUS_internal_error) {
    highsLogUser(options.log_options, HighsLogType::kError,
                 "Ipx: Internal error %" HIGHSINT_FORMAT "\n", (int)error_flag);
    return HighsStatus::kError;
  } else {
    highsLogUser(options.log_options, HighsLogType::kError,
                 "Ipx: unrecognised solve status = %" HIGHSINT_FORMAT "\n",
                 (int)solve_status);
    return HighsStatus::kError;
  }
  return HighsStatus::kError;
}

HighsStatus reportIpxIpmCrossoverStatus(const HighsOptions& options,
                                        const ipx::Int status,
                                        const bool ipm_status) {
  std::string method_name;
  if (ipm_status)
    method_name = "IPM      ";
  else
    method_name = "Crossover";
  if (status == IPX_STATUS_not_run) {
    if (ipm_status || options.run_crossover) {
      // Warn if method not run is IPM or run_crossover option is true
      highsLogUser(options.log_options, HighsLogType::kWarning,
		   "Ipx: %s not run\n", method_name.c_str());
      return HighsStatus::kWarning;
    }
    // OK if method not run is crossover and run_crossover option is
    // false!
    return HighsStatus::kOk;
  } else if (status == IPX_STATUS_optimal) {
    highsLogUser(options.log_options, HighsLogType::kInfo, "Ipx: %s optimal\n",
                 method_name.c_str());
    return HighsStatus::kOk;
  } else if (status == IPX_STATUS_imprecise) {
    highsLogUser(options.log_options, HighsLogType::kWarning,
                 "Ipx: %s imprecise\n", method_name.c_str());
    return HighsStatus::kWarning;
  } else if (status == IPX_STATUS_primal_infeas) {
    highsLogUser(options.log_options, HighsLogType::kWarning,
                 "Ipx: %s primal infeasible\n", method_name.c_str());
    return HighsStatus::kWarning;
  } else if (status == IPX_STATUS_dual_infeas) {
    highsLogUser(options.log_options, HighsLogType::kWarning,
                 "Ipx: %s dual infeasible\n", method_name.c_str());
    return HighsStatus::kWarning;
  } else if (status == IPX_STATUS_time_limit) {
    highsLogUser(options.log_options, HighsLogType::kWarning,
                 "Ipx: %s reached time limit\n", method_name.c_str());
    return HighsStatus::kWarning;
  } else if (status == IPX_STATUS_iter_limit) {
    highsLogUser(options.log_options, HighsLogType::kWarning,
                 "Ipx: %s reached iteration limit\n", method_name.c_str());
    return HighsStatus::kWarning;
  } else if (status == IPX_STATUS_no_progress) {
    highsLogUser(options.log_options, HighsLogType::kWarning,
                 "Ipx: %s no progress\n", method_name.c_str());
    return HighsStatus::kWarning;
  } else if (status == IPX_STATUS_failed) {
    highsLogUser(options.log_options, HighsLogType::kError, "Ipx: %s failed\n",
                 method_name.c_str());
    return HighsStatus::kError;
  } else if (status == IPX_STATUS_debug) {
    highsLogUser(options.log_options, HighsLogType::kError, "Ipx: %s debug\n",
                 method_name.c_str());
    return HighsStatus::kError;
  } else {
    highsLogUser(options.log_options, HighsLogType::kError,
                 "Ipx: %s unrecognised status\n", method_name.c_str());
    return HighsStatus::kError;
  }
  return HighsStatus::kError;
}

bool ipxStatusError(const bool status_error, const HighsOptions& options,
                    std::string message, const int value) {
  if (status_error) {
    if (value < 0) {
      highsLogUser(options.log_options, HighsLogType::kError, "Ipx: %s\n",
                   message.c_str());
    } else {
      highsLogUser(options.log_options, HighsLogType::kError, "Ipx: %s %d\n",
                   message.c_str(), value);
    }
    fflush(NULL);
  }
  assert(!status_error);
  return status_error;
}

bool illegalIpxSolvedStatus(const ipx::Info& ipx_info,
                            const HighsOptions& options) {
  bool found_illegal_status = false;
  //========
  // For IPX
  //========
  // Can solve and be optimal
  // Can solve and be imprecise
  // Can solve and be primal_infeas
  // Can solve and be dual_infeas
  // Cannot solve and reach time limit
  found_illegal_status =
      found_illegal_status ||
      ipxStatusError(ipx_info.status_ipm == IPX_STATUS_time_limit, options,
                     "solved  status_ipm should not be IPX_STATUS_time_limit");
  // Cannot solve and reach iteration limit
  found_illegal_status =
      found_illegal_status ||
      ipxStatusError(ipx_info.status_ipm == IPX_STATUS_iter_limit, options,
                     "solved  status_ipm should not be IPX_STATUS_iter_limit");
  // Cannot solve and make no progress
  found_illegal_status =
      found_illegal_status ||
      ipxStatusError(ipx_info.status_ipm == IPX_STATUS_no_progress, options,
                     "solved  status_ipm should not be IPX_STATUS_no_progress");
  // Cannot solve and failed
  found_illegal_status =
      found_illegal_status ||
      ipxStatusError(ipx_info.status_ipm == IPX_STATUS_failed, options,
                     "solved  status_ipm should not be IPX_STATUS_failed");
  // Cannot solve and debug
  found_illegal_status =
      found_illegal_status ||
      ipxStatusError(ipx_info.status_ipm == IPX_STATUS_debug, options,
                     "solved  status_ipm should not be IPX_STATUS_debug");
  //==============
  // For crossover
  //==============
  // Can solve and be optimal
  // Can solve and be imprecise
  // Cannot solve with primal infeasibility
  found_illegal_status =
      found_illegal_status ||
      ipxStatusError(
          ipx_info.status_crossover == IPX_STATUS_primal_infeas, options,
          "solved  status_crossover should not be IPX_STATUS_primal_infeas");
  // Cannot solve with dual infeasibility
  found_illegal_status =
      found_illegal_status ||
      ipxStatusError(
          ipx_info.status_crossover == IPX_STATUS_dual_infeas, options,
          "solved  status_crossover should not be IPX_STATUS_dual_infeas");
  // Cannot solve and reach time limit
  found_illegal_status =
      found_illegal_status ||
      ipxStatusError(
          ipx_info.status_crossover == IPX_STATUS_time_limit, options,
          "solved  status_crossover should not be IPX_STATUS_time_limit");
  // Cannot solve and reach time limit
  found_illegal_status =
      found_illegal_status ||
      ipxStatusError(
          ipx_info.status_crossover == IPX_STATUS_iter_limit, options,
          "solved  status_crossover should not be IPX_STATUS_iter_limit");
  // Cannot solve and make no progress
  found_illegal_status =
      found_illegal_status ||
      ipxStatusError(
          ipx_info.status_crossover == IPX_STATUS_no_progress, options,
          "solved  status_crossover should not be IPX_STATUS_no_progress");
  // Cannot solve and failed
  found_illegal_status =
      found_illegal_status ||
      ipxStatusError(
          ipx_info.status_crossover == IPX_STATUS_failed, options,
          "solved  status_crossover should not be IPX_STATUS_failed");
  // Cannot solve and debug
  found_illegal_status =
      found_illegal_status ||
      ipxStatusError(ipx_info.status_crossover == IPX_STATUS_debug, options,
                     "solved  status_crossover should not be IPX_STATUS_debug");
  return found_illegal_status;
}

bool illegalIpxStoppedIpmStatus(const ipx::Info& ipx_info,
                                const HighsOptions& options) {
  bool found_illegal_status = false;
  // Cannot stop and be optimal
  found_illegal_status =
      found_illegal_status ||
      ipxStatusError(ipx_info.status_ipm == IPX_STATUS_optimal, options,
                     "stopped status_ipm should not be IPX_STATUS_optimal");
  // Cannot stop and be imprecise
  found_illegal_status =
      found_illegal_status ||
      ipxStatusError(ipx_info.status_ipm == IPX_STATUS_imprecise, options,
                     "stopped status_ipm should not be IPX_STATUS_imprecise");
  // Cannot stop with primal infeasibility
  found_illegal_status =
      found_illegal_status ||
      ipxStatusError(
          ipx_info.status_ipm == IPX_STATUS_primal_infeas, options,
          "stopped status_ipm should not be IPX_STATUS_primal_infeas");
  // Cannot stop with dual infeasibility
  found_illegal_status =
      found_illegal_status ||
      ipxStatusError(ipx_info.status_ipm == IPX_STATUS_dual_infeas, options,
                     "stopped status_ipm should not be IPX_STATUS_dual_infeas");
  // Can stop with time limit
  // Can stop with iter limit
  // Can stop with no progress
  // Cannot stop and failed - should be error return earlier
  found_illegal_status =
      found_illegal_status ||
      ipxStatusError(ipx_info.status_ipm == IPX_STATUS_failed, options,
                     "stopped status_ipm should not be IPX_STATUS_failed");
  // Cannot stop and debug - should be error return earlier
  found_illegal_status =
      found_illegal_status ||
      ipxStatusError(ipx_info.status_ipm == IPX_STATUS_debug, options,
                     "stopped status_ipm should not be IPX_STATUS_debug");
  return found_illegal_status;
}

bool illegalIpxStoppedCrossoverStatus(const ipx::Info& ipx_info,
                                      const HighsOptions& options) {
  bool found_illegal_status = false;
  // Cannot stop and be optimal
  found_illegal_status =
      found_illegal_status ||
      ipxStatusError(
          ipx_info.status_crossover == IPX_STATUS_optimal, options,
          "stopped status_crossover should not be IPX_STATUS_optimal");
  // Cannot stop and be imprecise
  found_illegal_status =
      found_illegal_status ||
      ipxStatusError(
          ipx_info.status_crossover == IPX_STATUS_imprecise, options,
          "stopped status_crossover should not be IPX_STATUS_imprecise");
  // Cannot stop with primal infeasibility
  found_illegal_status =
      found_illegal_status ||
      ipxStatusError(
          ipx_info.status_crossover == IPX_STATUS_primal_infeas, options,
          "stopped status_crossover should not be IPX_STATUS_primal_infeas");
  // Cannot stop with dual infeasibility
  found_illegal_status =
      found_illegal_status ||
      ipxStatusError(
          ipx_info.status_crossover == IPX_STATUS_dual_infeas, options,
          "stopped status_crossover should not be IPX_STATUS_dual_infeas");
  // Cannot stop and reach iteration limit
  found_illegal_status =
      found_illegal_status ||
      ipxStatusError(
          ipx_info.status_crossover == IPX_STATUS_iter_limit, options,
          "stopped status_crossover should not be IPX_STATUS_iter_limit");
  // Can stop and reach time limit
  // Cannot stop with no_progress
  found_illegal_status =
      found_illegal_status ||
      ipxStatusError(
          ipx_info.status_crossover == IPX_STATUS_no_progress, options,
          "stopped status_crossover should not be IPX_STATUS_no_progress");
  // Cannot stop and failed - should be error return earlier
  found_illegal_status =
      found_illegal_status ||
      ipxStatusError(
          ipx_info.status_crossover == IPX_STATUS_failed, options,
          "stopped status_crossover should not be IPX_STATUS_failed");
  // Cannot stop and debug - should be error return earlier
  found_illegal_status =
      found_illegal_status ||
      ipxStatusError(ipx_info.status_crossover == IPX_STATUS_debug, options,
                     "stopped status_crossover should not be IPX_STATUS_debug");
  return found_illegal_status;
}

void reportIpmNoProgress(const HighsOptions& options,
                         const ipx::Info& ipx_info) {
  highsLogUser(options.log_options, HighsLogType::kWarning,
               "No progress: primal objective value       = %11.4g\n",
               ipx_info.pobjval);
  highsLogUser(options.log_options, HighsLogType::kWarning,
               "No progress: max absolute primal residual = %11.4g\n",
               ipx_info.abs_presidual);
  highsLogUser(options.log_options, HighsLogType::kWarning,
               "No progress: max absolute   dual residual = %11.4g\n",
               ipx_info.abs_dresidual);
}

void getHighsNonVertexSolution(const HighsOptions& options,
                               const HighsLp& lp, const ipx::Int num_col,
                               const ipx::Int num_row,
                               const std::vector<double>& rhs,
                               const std::vector<char>& constraint_type,
                               const ipx::LpSolver& lps,
			       const HighsModelStatus model_status,
                               HighsSolution& highs_solution) {
  // Get the interior solution (available if IPM was started).
  // GetInteriorSolution() returns the final IPM iterate, regardless if the
  // IPM terminated successfully or not. (Only in case of out-of-memory no
  // solution exists.)
  std::vector<double> x(num_col);
  std::vector<double> xl(num_col);
  std::vector<double> xu(num_col);
  std::vector<double> zl(num_col);
  std::vector<double> zu(num_col);
  std::vector<double> slack(num_row);
  std::vector<double> y(num_row);

  lps.GetInteriorSolution(&x[0], &xl[0], &xu[0], &slack[0], &y[0], &zl[0],
                          &zu[0]);

  ipxSolutionToHighsSolution(options, lp, rhs, constraint_type, num_col,
                             num_row, x, slack, y, zl, zu,
			     model_status, highs_solution);
}

void reportSolveData(const HighsLogOptions& log_options, const ipx::Info& ipx_info) {
  highsLogDev(log_options, HighsLogType::kInfo, "\nIPX Solve data\n");
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    IPX       status = %4d\n", (int)ipx_info.status);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    IPM       status = %4d\n", (int)ipx_info.status_ipm);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Crossover status = %4d\n", (int)ipx_info.status_crossover);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    IPX errflag      = %4d\n\n", (int)ipx_info.errflag);

  highsLogDev(log_options, HighsLogType::kInfo,
	 "    LP variables   = %8d\n", (int)ipx_info.num_var);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    LP constraints = %8d\n", (int)ipx_info.num_constr);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    LP entries     = %8d\n\n", (int)ipx_info.num_entries);

  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Solver columns = %8d\n", (int)ipx_info.num_cols_solver);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Solver rows    = %8d\n", (int)ipx_info.num_rows_solver);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Solver entries = %8d\n\n", (int)ipx_info.num_entries_solver);

  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Dualized = %d\n", (int)ipx_info.dualized);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Number of dense columns detected = %d\n\n", (int)ipx_info.dense_cols);

  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Dependent rows    = %d\n", (int)ipx_info.dependent_rows);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Dependent cols    = %d\n", (int)ipx_info.dependent_cols);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Inconsistent rows = %d\n", (int)ipx_info.rows_inconsistent);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Inconsistent cols = %d\n", (int)ipx_info.cols_inconsistent);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Primal dropped    = %d\n", (int)ipx_info.primal_dropped);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Dual   dropped    = %d\n\n", (int)ipx_info.dual_dropped);

  highsLogDev(log_options, HighsLogType::kInfo,
	 "    |Absolute primal residual| = %11.4g\n", ipx_info.abs_presidual);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    |Absolute   dual residual| = %11.4g\n", ipx_info.abs_dresidual);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    |Relative primal residual| = %11.4g\n", ipx_info.rel_presidual);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    |Relative   dual residual| = %11.4g\n\n", ipx_info.rel_dresidual);

  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Primal objective value     = %11.4g\n", ipx_info.pobjval);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Dual   objective value     = %11.4g\n", ipx_info.dobjval);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Relative objective gap     = %11.4g\n", ipx_info.rel_objgap);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Complementarity            = %11.4g\n\n", ipx_info.complementarity);

  highsLogDev(log_options, HighsLogType::kInfo,
	 "    |x| = %11.4g\n", ipx_info.normx);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    |y| = %11.4g\n", ipx_info.normy);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    |z| = %11.4g\n\n", ipx_info.normz);

  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Objective value       = %11.4g\n", ipx_info.objval);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Primal infeasiblility = %11.4g\n", ipx_info.primal_infeas);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Dual infeasiblility   = %11.4g\n\n", ipx_info.dual_infeas);
  
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    IPM iter   = %d\n", (int)ipx_info.iter);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    KKT iter 1 = %d\n", (int)ipx_info.kktiter1);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    KKT iter 2 = %d\n", (int)ipx_info.kktiter2);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Basis repairs = %d\n", (int)ipx_info.basis_repairs);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Updates start     = %d\n", (int)ipx_info.updates_start);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Updates ipm       = %d\n", (int)ipx_info.updates_ipm);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Updates crossover = %d\n\n", (int)ipx_info.updates_crossover);

  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Time total          = %8.2f\n\n", ipx_info.time_total);
  double sum_time = 0;
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Time IPM 1          = %8.2f\n", ipx_info.time_ipm1);
  sum_time += ipx_info.time_ipm1;
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Time IPM 2          = %8.2f\n", ipx_info.time_ipm2);
  sum_time += ipx_info.time_ipm2;
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Time starting basis = %8.2f\n", ipx_info.time_starting_basis);
  sum_time += ipx_info.time_starting_basis;
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Time crossover      = %8.2f\n", ipx_info.time_crossover);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Sum                 = %8.2f\n\n", sum_time);

  sum_time = 0;
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Time kkt_factorize  = %8.2f\n", ipx_info.time_kkt_factorize);
  sum_time += ipx_info.time_kkt_factorize;
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Time kkt_solve      = %8.2f\n", ipx_info.time_kkt_solve);
  sum_time += ipx_info.time_kkt_solve;
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Sum                 = %8.2f\n\n", sum_time);

  sum_time = 0;
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Time maxvol         = %8.2f\n", ipx_info.time_maxvol);
  sum_time += ipx_info.time_maxvol;
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Time cr1            = %8.2f\n", ipx_info.time_cr1);
  sum_time += ipx_info.time_cr1;
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Time cr2            = %8.2f\n", ipx_info.time_cr2);
  sum_time += ipx_info.time_cr2;
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Sum                 = %8.2f\n\n", sum_time);

  sum_time = 0;
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Time cr1_AAt        = %8.2f\n", ipx_info.time_cr1_AAt);
  sum_time += ipx_info.time_cr1_AAt;
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Time cr1_pre        = %8.2f\n", ipx_info.time_cr1_pre);
  sum_time += ipx_info.time_cr1_pre;
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Sum  cr1            = %8.2f\n\n", sum_time);

  sum_time = 0;
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Time cr2_NNt        = %8.2f\n", ipx_info.time_cr2_NNt);
  sum_time += ipx_info.time_cr2_NNt;
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Time cr2_B          = %8.2f\n", ipx_info.time_cr2_B);
  sum_time += ipx_info.time_cr2_B;
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Time cr2_Bt         = %8.2f\n", ipx_info.time_cr2_Bt);
  sum_time += ipx_info.time_cr2_Bt;
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Sum  cr2            = %8.2f\n\n", sum_time);

  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Proportion of sparse FTRAN = %11.4g\n", ipx_info.ftran_sparse);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Proportion of sparse BTRAN = %11.4g\n\n", ipx_info.btran_sparse);

  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Time FTRAN       = %8.2f\n", ipx_info.time_ftran);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Time BTRAN       = %8.2f\n", ipx_info.time_btran);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Time LU INVERT   = %8.2f\n", ipx_info.time_lu_invert);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Time LU UPDATE   = %8.2f\n", ipx_info.time_lu_update);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Mean fill-in     = %11.4g\n", ipx_info.mean_fill);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Max fill-in      = %11.4g\n", ipx_info.max_fill);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Time symb INVERT = %11.4g\n\n", ipx_info.time_symb_invert);
  
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Maxvol updates       = %d\n", (int)ipx_info.maxvol_updates);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Maxvol skipped       = %d\n", (int)ipx_info.maxvol_skipped);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Maxvol passes        = %d\n", (int)ipx_info.maxvol_passes);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Tableau num nonzeros = %d\n", (int)ipx_info.tbl_nnz);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Tbl max?             = %11.4g\n", ipx_info.tbl_max);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Frobnorm squared     = %11.4g\n", ipx_info.frobnorm_squared);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Lambda max           = %11.4g\n", ipx_info.lambdamax);
  highsLogDev(log_options, HighsLogType::kInfo,
	 "    Volume increase      = %11.4g\n\n", ipx_info.volume_increase);

}
