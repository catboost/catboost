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
#ifndef SIMPLEX_HAPP_H_
#define SIMPLEX_HAPP_H_

// todo: clear includes.
//#include <cstring>
//#include <fstream>
//#include <iomanip>
//#include <iostream>
//#include <map>
//#include <set>
//#include <vector>

#include "lp_data/HighsLpSolverObject.h"
#include "lp_data/HighsLpUtils.h"
#include "lp_data/HighsSolution.h"
#include "lp_data/HighsSolve.h"
#include "simplex/HEkk.h"
#include "simplex/HSimplex.h"

// Single method to solve an LP with the simplex method. Solves the
// scaled LP then analyses the unscaled solution. If it doesn't satisfy
// the required tolerances, tolerances for the scaled LP are
// identified which, if used, might yield an unscaled solution that
// satisfies the required tolerances.
//
// If possible, it sets the HiGHS basis and solution

inline HighsStatus returnFromSolveLpSimplex(HighsLpSolverObject& solver_object,
                                            HighsStatus return_status) {
  HighsOptions& options = solver_object.options_;
  HEkk& ekk_instance = solver_object.ekk_instance_;
  HighsLp& incumbent_lp = solver_object.lp_;
  // Copy the simplex iteration count to highs_info_ from ekk_instance
  solver_object.highs_info_.simplex_iteration_count =
      ekk_instance.iteration_count_;
  // Ensure that the incumbent LP is neither moved, nor scaled
  assert(!incumbent_lp.is_moved_);
  assert(!incumbent_lp.is_scaled_);
  // Cannot expect any more with an error return, and safer to clear
  // HEkk than try to retain any data
  if (return_status == HighsStatus::kError) {
    ekk_instance.clear();
    return return_status;
  }
  //
  // Ensure that there is an invert for the current LP
  assert(ekk_instance.status_.has_invert);
  // Ensure that simplex NLA is set up
  assert(ekk_instance.status_.has_nla);
  // Set the simplex NLA scaling
  ekk_instance.setNlaPointersForLpAndScale(incumbent_lp);
  assert(ekk_instance.debugNlaScalingOk(incumbent_lp));
  HighsInt alt_debug_level = -1;
  // Forced expensive debug for development work
  //  alt_debug_level = kHighsDebugLevelExpensive;
  if (ekk_instance.debugNlaCheckInvert("HApp: returnFromSolveLpSimplex",
                                       alt_debug_level) ==
      HighsDebugStatus::kError) {
    highsLogUser(options.log_options, HighsLogType::kError,
                 "Error in basis matrix inverse after solving the LP\n");
    return_status = HighsStatus::kError;
  }
  return return_status;
}

inline HighsStatus solveLpSimplex(HighsLpSolverObject& solver_object) {
  HighsStatus return_status = HighsStatus::kOk;
  HighsStatus call_status;
  HighsOptions& options = solver_object.options_;
  HighsLp& incumbent_lp = solver_object.lp_;
  HighsSolution& solution = solver_object.solution_;
  HighsModelStatus& model_status = solver_object.model_status_;
  HighsModelStatus scaled_model_status = HighsModelStatus::kUnknown;
  HighsInfo& highs_info = solver_object.highs_info_;
  HighsBasis& basis = solver_object.basis_;

  HEkk& ekk_instance = solver_object.ekk_instance_;
  HighsLp& ekk_lp = ekk_instance.lp_;
  HighsSimplexInfo& ekk_info = ekk_instance.info_;
  SimplexBasis& ekk_basis = ekk_instance.basis_;
  HighsSimplexStatus& status = ekk_instance.status_;

  // Check that any retained Ekk data - basis and NLA - are OK on entry
  bool retained_ekk_data_ok = ekk_instance.debugRetainedDataOk(incumbent_lp) !=
                              HighsDebugStatus::kLogicalError;
  if (!retained_ekk_data_ok) {
    highsLogUser(options.log_options, HighsLogType::kError,
                 "solveLpSimplex: Retained Ekk data not OK on entry\n");
    assert(retained_ekk_data_ok);
    return_status = HighsStatus::kError;
  }

  // Copy the simplex iteration count from highs_info_ to ekk_instance, just for
  // convenience
  ekk_instance.iteration_count_ = highs_info.simplex_iteration_count;

  // Reset the model status and HighsInfo values in case of premature
  // return
  resetModelStatusAndHighsInfo(solver_object);

  // Assumes that the LP has a positive number of rows, since
  // unconstrained LPs should be solved in solveLp
  bool positive_num_row = solver_object.lp_.num_row_ > 0;
  assert(positive_num_row);
  if (!positive_num_row) {
    highsLogUser(
        options.log_options, HighsLogType::kError,
        "solveLpSimplex called for LP with non-positive (%" HIGHSINT_FORMAT
        ") "
        "number of constraints\n",
        incumbent_lp.num_row_);
    return returnFromSolveLpSimplex(solver_object, HighsStatus::kError);
  }
  // On entry to solveLpSimplex, the incumbent LP is assumed to be
  // unscaled and not moved
  assert(!incumbent_lp.is_scaled_);
  assert(!incumbent_lp.is_moved_);
  // Consider scaling the LP - either with any existing scaling, or by
  // considering computing scaling factors if there are none - and
  // then move to EKK
  const bool new_scaling = considerScaling(options, incumbent_lp);
  // If new scaling is performed, the hot start information is
  // no longer valid
  if (new_scaling) ekk_instance.clearHotStart();
  // Move the LP to EKK, updating other EKK pointers and any simplex
  // NLA pointers, since they may have moved if the LP has been
  // modified
  ekk_instance.moveLp(solver_object);
  if (!status.has_basis) {
    // There is no simplex basis, so use any HiGHS basis
    if (basis.valid) {
      call_status = ekk_instance.setBasis(basis);
      if (call_status == HighsStatus::kError) {
        incumbent_lp.moveBackLpAndUnapplyScaling(ekk_lp);
        return returnFromSolveLpSimplex(solver_object, call_status);
      }
    } else {
      // Starting from a logical basis, so consider dualising and/or
      // permuting the LP
      if (options.simplex_dualise_strategy == kHighsOptionChoose ||
          options.simplex_dualise_strategy == kHighsOptionOn) {
        // Dualise unless we choose not to
        bool dualise_lp = true;
        if (options.simplex_dualise_strategy == kHighsOptionChoose) {
          if (incumbent_lp.num_row_ < 10 * incumbent_lp.num_col_)
            dualise_lp = false;
        }
        if (dualise_lp) ekk_instance.dualise();
      }
      if (options.simplex_permute_strategy == kHighsOptionChoose ||
          options.simplex_permute_strategy == kHighsOptionOn) {
        // Permute the LP
        ekk_instance.permute();
      }
    }
  }
  // These local illegal values are over-written with correct values
  // if the scaled LP is solved in order to take correct algorithmic
  // decisions if the unscaled LP is solved later
  HighsInt num_unscaled_primal_infeasibilities =
      kHighsIllegalInfeasibilityCount;
  HighsInt num_unscaled_dual_infeasibilities = kHighsIllegalInfeasibilityCount;
  // Record whether the unscaled LP is to be solved, and has been
  // solved. It's not solved if it's proved to be infeasible using the
  // ray from the scaled LP, in which case the solution (FWIW) and
  // basis are taken from the unscaled solution of the scaled LP.
  bool solve_unscaled_lp = false;
  bool solved_unscaled_lp = false;
  if (!incumbent_lp.scale_.has_scaling) {
    //
    // Solve the unscaled LP with unscaled NLA
    //
    solve_unscaled_lp = true;
    return_status = ekk_instance.solve();
    solved_unscaled_lp = true;
    ekk_instance.unpermute();
    ekk_instance.undualise();
    assert(!ekk_instance.status_.is_permuted &&
           !ekk_instance.status_.is_dualised);
    if (options.cost_scale_factor) {
      double cost_scale_factor = pow(2.0, -options.cost_scale_factor);
      highsLogDev(options.log_options, HighsLogType::kInfo,
                  "Objective = %11.4g\n",
                  cost_scale_factor * ekk_instance.info_.dual_objective_value);
      ekk_instance.model_status_ = HighsModelStatus::kNotset;
      return_status = HighsStatus::kError;
    }
    //
  } else {
    // Indicate that there is no (current) need to refine the solution
    // by solving the unscaled LP with scaled NLA
    bool refine_solution = false;
    if (options.simplex_unscaled_solution_strategy ==
            kSimplexUnscaledSolutionStrategyNone ||
        options.simplex_unscaled_solution_strategy ==
            kSimplexUnscaledSolutionStrategyRefine) {
      //
      // Solve the scaled LP!
      //
      return_status = ekk_instance.solve();
      ekk_instance.unpermute();
      ekk_instance.undualise();
      assert(!ekk_instance.status_.is_permuted &&
             !ekk_instance.status_.is_dualised);
      //
      if (options.cost_scale_factor) {
        double cost_scale_factor = pow(2.0, -options.cost_scale_factor);
        highsLogDev(
            options.log_options, HighsLogType::kInfo, "Objective = %11.4g\n",
            cost_scale_factor * ekk_instance.info_.dual_objective_value);
        ekk_instance.model_status_ = HighsModelStatus::kNotset;
        return_status = HighsStatus::kError;
      }
      if (return_status == HighsStatus::kError) {
        incumbent_lp.moveBackLpAndUnapplyScaling(ekk_lp);
        return returnFromSolveLpSimplex(solver_object, return_status);
      }
      // Copy solution data from the EKK instance
      scaled_model_status = ekk_instance.model_status_;
      highs_info.objective_function_value = ekk_info.primal_objective_value;
      highs_info.simplex_iteration_count = ekk_instance.iteration_count_;
      solution = ekk_instance.getSolution();
      basis = ekk_instance.getHighsBasis(ekk_lp);
      assert(basis.valid);
      highs_info.basis_validity = kBasisValidityValid;
      incumbent_lp.moveBackLpAndUnapplyScaling(ekk_lp);
      // Now that the incumbent LP is unscaled, to use the simplex NLA
      // requires scaling to be applied
      ekk_instance.setNlaPointersForLpAndScale(incumbent_lp);
      unscaleSolution(solution, incumbent_lp.scale_);
      // Determine whether the unscaled LP has been solved
      getUnscaledInfeasibilities(options, incumbent_lp.scale_, ekk_basis,
                                 ekk_info, highs_info);
      num_unscaled_primal_infeasibilities =
          highs_info.num_primal_infeasibilities;
      num_unscaled_dual_infeasibilities = highs_info.num_dual_infeasibilities;
      // Determine whether the unscaled solution has infeasibilities
      // after the scaled LP has been solved to optimality
      const bool scaled_optimality_but_unscaled_infeasibilities =
          scaled_model_status == HighsModelStatus::kOptimal &&
          (num_unscaled_primal_infeasibilities ||
           num_unscaled_dual_infeasibilities);
      if (scaled_optimality_but_unscaled_infeasibilities)
        highsLogDev(options.log_options, HighsLogType::kInfo,
                    "Have num/max/sum primal (%" HIGHSINT_FORMAT
                    "/%g/%g) and dual (%" HIGHSINT_FORMAT
                    "/%g/%g) "
                    "unscaled infeasibilities\n",
                    highs_info.num_primal_infeasibilities,
                    highs_info.max_primal_infeasibility,
                    highs_info.sum_primal_infeasibilities,
                    highs_info.num_dual_infeasibilities,
                    highs_info.max_dual_infeasibility,
                    highs_info.sum_dual_infeasibilities);
      // Determine whether refinement will take place
      refine_solution =
          options.simplex_unscaled_solution_strategy ==
              kSimplexUnscaledSolutionStrategyRefine &&
          (scaled_optimality_but_unscaled_infeasibilities ||
           scaled_model_status == HighsModelStatus::kInfeasible ||
           scaled_model_status == HighsModelStatus::kUnboundedOrInfeasible ||
           scaled_model_status == HighsModelStatus::kUnbounded ||
           scaled_model_status == HighsModelStatus::kObjectiveBound ||
           scaled_model_status == HighsModelStatus::kObjectiveTarget ||
           scaled_model_status == HighsModelStatus::kUnknown);
      // Handle the case when refinement will not take place
      if (!refine_solution) {
        model_status = scaled_model_status;
        return_status = highsStatusFromHighsModelStatus(model_status);
        return returnFromSolveLpSimplex(solver_object, return_status);
      }
    } else {
      // LP is scaled, but simplex_unscaled_solution_strategy is
      // kSimplexUnscaledSolutionStrategyDirect, so have to move back
      // the LP and unscale it
      assert(options.simplex_unscaled_solution_strategy ==
             kSimplexUnscaledSolutionStrategyDirect);
      incumbent_lp.moveBackLpAndUnapplyScaling(ekk_lp);
    }
    assert(options.simplex_unscaled_solution_strategy ==
               kSimplexUnscaledSolutionStrategyDirect ||
           refine_solution);
    // Solve the unscaled LP using scaled NLA. This requires pointers of
    // a scaled matrix to be passed to the HFactor instance. Use the
    // incumbent LP for this.
    //
    // Check that the incumbent LP has been moved back and is unscaled
    assert(!incumbent_lp.is_moved_);
    assert(!incumbent_lp.is_scaled_);
    // Move the incumbent LP
    ekk_instance.moveLp(solver_object);
    // If refining after proving primal infeasibility of the scaled
    // LP, see whether the proof still holds for the unscaled LP. If
    // it does, then there's no need to solve the unscaled LP
    solve_unscaled_lp = true;
    // ToDo: ekk_instance.status_.has_dual_ray should now be true if
    // scaled_model_status == HighsModelStatus::kInfeasible since this
    // model status depends on the infeasibility proof being true
    if (scaled_model_status == HighsModelStatus::kInfeasible &&
        ekk_instance.status_.has_dual_ray) {
      ekk_instance.setNlaPointersForLpAndScale(ekk_lp);
      if (ekk_instance.proofOfPrimalInfeasibility()) solve_unscaled_lp = false;
    }
    if (solve_unscaled_lp) {
      // Save options/strategies that may be changed
      HighsInt simplex_strategy = options.simplex_strategy;
      double dual_simplex_cost_perturbation_multiplier =
          options.dual_simplex_cost_perturbation_multiplier;
      HighsInt simplex_dual_edge_weight_strategy =
          ekk_info.dual_edge_weight_strategy;
      if (num_unscaled_primal_infeasibilities == 0 ||
          scaled_model_status == HighsModelStatus::kObjectiveBound) {
        // Only dual infeasibilities, or primal infeasibilities do not
        // matter due to solution status, so use primal simplex phase
        // 2
        options.simplex_strategy = kSimplexStrategyPrimal;
        if (scaled_model_status == HighsModelStatus::kObjectiveBound) {
          highsLogDev(
              options.log_options, HighsLogType::kInfo,
              "solveLpSimplex: Calling primal simplex after "
              "scaled_model_status == HighsModelStatus::kObjectiveBound: solve "
              "= %d; tick = %d; iter = %d\n",
              (int)ekk_instance.debug_solve_call_num_,
              (int)ekk_instance.debug_initial_build_synthetic_tick_,
              (int)ekk_instance.iteration_count_);
        }
      } else {
        // Using dual simplex, so force Devex if starting from an advanced
        // basis with no steepest edge weights
        if ((status.has_basis || basis.valid) &&
            !status.has_dual_steepest_edge_weights) {
          ekk_info.dual_edge_weight_strategy = kSimplexEdgeWeightStrategyDevex;
        }
      }
      //
      // Solve the unscaled LP with scaled NLA
      //
      // Force the simplex solver to start in phase 2 unless solving
      // the LP directly as unscaled
      //
      const bool force_phase2 = options.simplex_unscaled_solution_strategy !=
                                kSimplexUnscaledSolutionStrategyDirect;
      return_status = ekk_instance.solve(force_phase2);
      solved_unscaled_lp = true;
      if (scaled_model_status != HighsModelStatus::kObjectiveBound &&
          ekk_instance.model_status_ == HighsModelStatus::kObjectiveBound) {
        // it may happen that the unscaled LP detected status kObjectiveBound
        // for the first time in which case we again call solve with primal
        // simplex if not dual feasible
        const bool objective_bound_refinement =
            ekk_instance.model_status_ == HighsModelStatus::kObjectiveBound &&
            ekk_info.num_dual_infeasibilities > 0;
        if (objective_bound_refinement) {
          options.simplex_strategy = kSimplexStrategyPrimal;
          return_status = ekk_instance.solve(force_phase2);
        }
      }
      // Restore the options/strategies that may have been changed
      options.simplex_strategy = simplex_strategy;
      options.dual_simplex_cost_perturbation_multiplier =
          dual_simplex_cost_perturbation_multiplier;
      ekk_info.dual_edge_weight_strategy = simplex_dual_edge_weight_strategy;
    }
  }
  if (solved_unscaled_lp) {
    // Copy solution data from the EKK istance
    scaled_model_status = ekk_instance.model_status_;
    highs_info.objective_function_value = ekk_info.primal_objective_value;
    highs_info.simplex_iteration_count = ekk_instance.iteration_count_;
    solution = ekk_instance.getSolution();
    basis = ekk_instance.getHighsBasis(ekk_lp);
    assert(basis.valid);
    highs_info.basis_validity = kBasisValidityValid;
  }
  // Move the incumbent LP back from Ekk
  incumbent_lp = std::move(ekk_lp);
  incumbent_lp.is_moved_ = false;
  ekk_instance.setNlaPointersForLpAndScale(incumbent_lp);
  if (return_status == HighsStatus::kError) {
    // Error return, so make sure that the unscaled and scaled model
    // status values are the same, and that they correspond to an
    // error return
    model_status = scaled_model_status;
    return_status = highsStatusFromHighsModelStatus(model_status);
    assert(return_status == HighsStatus::kError);
    return returnFromSolveLpSimplex(solver_object, HighsStatus::kError);
  }
  if (solved_unscaled_lp) {
    // The unscaled LP has been solved - either directly, or because
    // there was no scaling. Copy values into HighsInfo from solving
    // the scaled LP.
    assert(solve_unscaled_lp);
    highs_info.num_primal_infeasibilities = ekk_info.num_primal_infeasibilities;
    highs_info.max_primal_infeasibility = ekk_info.max_primal_infeasibility;
    highs_info.sum_primal_infeasibilities = ekk_info.sum_primal_infeasibilities;
    highs_info.num_dual_infeasibilities = ekk_info.num_dual_infeasibilities;
    highs_info.max_dual_infeasibility = ekk_info.max_dual_infeasibility;
    highs_info.sum_dual_infeasibilities = ekk_info.sum_dual_infeasibilities;
  } else {
    // The unscaled LP has not been solved because the scaled LP was
    // infeasible and the proof of infeasibility held for the unscaled
    // LP. Hence the values in HighsInfo that are set (above) by the
    // call to getUnscaledInfeasibilities are correct
    assert(!solve_unscaled_lp);
    assert(scaled_model_status == HighsModelStatus::kInfeasible);
  }
  setSolutionStatus(highs_info);
  model_status = scaled_model_status;
  return_status = highsStatusFromHighsModelStatus(model_status);
  return returnFromSolveLpSimplex(solver_object, return_status);
}
#endif
