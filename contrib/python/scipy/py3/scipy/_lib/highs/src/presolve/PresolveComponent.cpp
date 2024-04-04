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
/**@file PresolveComponent.cpp
 * @brief The HiGHS class
 */

#include "presolve/PresolveComponent.h"

#include "presolve/HPresolve.h"

HighsStatus PresolveComponent::init(const HighsLp& lp, HighsTimer& timer,
                                    bool mip) {
  data_.postSolveStack.initializeIndexMaps(lp.num_row_, lp.num_col_);
  data_.reduced_lp_ = lp;
  this->timer = &timer;
  return HighsStatus::kOk;
}

HighsStatus PresolveComponent::setOptions(const HighsOptions& options) {
  options_ = &options;

  return HighsStatus::kOk;
}

std::string PresolveComponent::presolveStatusToString(
    const HighsPresolveStatus presolve_status) {
  switch (presolve_status) {
    case HighsPresolveStatus::kNotPresolved:
      return "Not presolved";
    case HighsPresolveStatus::kNotReduced:
      return "Not reduced";
    case HighsPresolveStatus::kInfeasible:
      return "Infeasible";
    case HighsPresolveStatus::kUnboundedOrInfeasible:
      return "Unbounded or infeasible";
    case HighsPresolveStatus::kReduced:
      return "Reduced";
    case HighsPresolveStatus::kReducedToEmpty:
      return "Reduced to empty";
    case HighsPresolveStatus::kTimeout:
      return "Timeout";
    case HighsPresolveStatus::kNullError:
      return "Null error";
    case HighsPresolveStatus::kOptionsError:
      return "Options error";
    default:
      assert(1 == 0);
      return "Unrecognised presolve status";
  }
}

void PresolveComponent::negateReducedLpColDuals(bool reduced) {
  for (HighsInt col = 0; col < data_.reduced_lp_.num_col_; col++)
    data_.recovered_solution_.col_dual[col] =
        -data_.recovered_solution_.col_dual[col];
  return;
}

void PresolveComponent::negateReducedLpCost() { return; }

HighsPresolveStatus PresolveComponent::run() {
  presolve::HPresolve presolve;
  presolve.setInput(data_.reduced_lp_, *options_, timer);

  HighsModelStatus status = presolve.run(data_.postSolveStack);

  // Ensure that the presolve status is used to set
  // presolve_.presolve_status_, as well as being returned
  HighsPresolveStatus presolve_status;
  switch (status) {
    case HighsModelStatus::kInfeasible:
      presolve_status = HighsPresolveStatus::kInfeasible;
      break;
    case HighsModelStatus::kUnboundedOrInfeasible:
      presolve_status = HighsPresolveStatus::kUnboundedOrInfeasible;
      break;
    case HighsModelStatus::kOptimal:
      presolve_status = HighsPresolveStatus::kReducedToEmpty;
      break;
    default:
      if (data_.postSolveStack.numReductions() == 0)
        presolve_status = HighsPresolveStatus::kNotReduced;
      else
        presolve_status = HighsPresolveStatus::kReduced;
  }
  this->presolve_status_ = presolve_status;
  return presolve_status;
}

void PresolveComponent::clear() {
  //  has_run_ = false;
  data_.clear();
}
namespace presolve {

bool checkOptions(const PresolveComponentOptions& options) {
  // todo: check options in a smart way
  if (options.dev) std::cout << "Checking presolve options... ";

  if (!(options.iteration_strategy == "smart" ||
        options.iteration_strategy == "off" ||
        options.iteration_strategy == "num_limit")) {
    if (options.dev)
      std::cout << "error: iteration strategy unknown: "
                << options.iteration_strategy << "." << std::endl;
    return false;
  }

  if (options.iteration_strategy == "num_limit" && options.max_iterations < 0) {
    if (options.dev)
      std::cout << "warning: negative iteration limit: "
                << options.max_iterations
                << ". Presolve will be run with no limit on iterations."
                << std::endl;
    return false;
  }

  return true;
}

}  // namespace presolve
