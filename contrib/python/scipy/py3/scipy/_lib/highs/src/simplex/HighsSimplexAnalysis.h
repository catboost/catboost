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
/**@file simplex/HighsSimplexAnalysis.h
 * @brief Analyse simplex iterations, both for run-time control and data
 * gathering
 */
#ifndef SIMPLEX_HIGHSSIMPLEXANALYSIS_H_
#define SIMPLEX_HIGHSSIMPLEXANALYSIS_H_

#include <cassert>
#include <memory>
#include <sstream>

#include "lp_data/HighsLp.h"
#include "lp_data/HighsOptions.h"
#include "simplex/SimplexConst.h"
#include "util/HFactor.h"
#include "util/HVector.h"
#include "util/HVectorBase.h"
#include "util/HighsInt.h"
#include "util/HighsTimer.h"
#include "util/HighsUtils.h"

enum TRAN_STAGE {
  TRAN_STAGE_FTRAN_LOWER = 0,
  TRAN_STAGE_FTRAN_UPPER_FT,
  TRAN_STAGE_FTRAN_UPPER,
  TRAN_STAGE_BTRAN_UPPER,
  TRAN_STAGE_BTRAN_UPPER_FT,
  TRAN_STAGE_BTRAN_LOWER,
  NUM_TRAN_STAGE_TYPE,
};

struct TranStageAnalysis {
  std::string name_;
  HighsScatterData rhs_density_;
  HighsInt num_decision_;
  HighsInt num_wrong_original_sparse_decision_;
  HighsInt num_wrong_original_hyper_decision_;
  HighsInt num_wrong_new_sparse_decision_;
  HighsInt num_wrong_new_hyper_decision_;
};

const HighsInt kAnIterTraceMaxNumRec = 20;
const HighsLogType kIterationReportLogType = HighsLogType::kVerbose;

/**
 * @brief Analyse simplex iterations, both for run-time control and data
 * gathering
 */
class HighsSimplexAnalysis {
 public:
  HighsSimplexAnalysis() {}
  // Pointer to timer
  HighsTimer* timer_;

  void setup(const std::string lp_name, const HighsLp& lp,
             const HighsOptions& options,
             const HighsInt simplex_iteration_count);
  void setupSimplexTime(const HighsOptions& options);
  void setupFactorTime(const HighsOptions& options);
  void messaging(const HighsLogOptions& log_options_);
  void iterationReport();
  void invertReport();
  void invertReport(const bool header);
  void userInvertReport(const bool force);
  void userInvertReport(const bool header, const bool force);
  bool predictEndDensity(const HighsInt tran_stage_id,
                         const double start_density, double& end_density);
  void afterTranStage(const HighsInt tran_stage_id, const double start_density,
                      const double end_density, const double historical_density,
                      const double predicted_end_density,
                      const bool use_solve_sparse_original_HFactor_logic,
                      const bool use_solve_sparse_new_HFactor_logic);

  void simplexTimerStart(const HighsInt simplex_clock,
                         const HighsInt thread_id = 0);
  void simplexTimerStop(const HighsInt simplex_clock,
                        const HighsInt thread_id = 0);
  bool simplexTimerRunning(const HighsInt simplex_clock,
                           const HighsInt thread_id = 0);
  HighsInt simplexTimerNumCall(const HighsInt simplex_clock,
                               const HighsInt thread_id = 0);
  double simplexTimerRead(const HighsInt simplex_clock,
                          const HighsInt thread_id = 0);

  HighsTimerClock* getThreadFactorTimerClockPointer();

  const std::vector<HighsTimerClock>& getThreadSimplexTimerClocks() {
    return thread_simplex_clocks;
  }
  HighsTimerClock* getThreadSimplexTimerClockPtr(HighsInt i) {
    assert(i >= 0 && i < (HighsInt)thread_simplex_clocks.size());
    return &thread_simplex_clocks[i];
  }

  const std::vector<HighsTimerClock>& getThreadFactorTimerClocks() {
    return thread_factor_clocks;
  }
  HighsTimerClock* getThreadFactorTimerClockPtr(HighsInt i) {
    assert(i >= 0 && i < (HighsInt)thread_factor_clocks.size());
    return &thread_factor_clocks[i];
  }

  void iterationRecord();
  void iterationRecordMajor();
  void operationRecordBefore(const HighsInt operation_type,
                             const HVector& vector,
                             const double historical_density);
  void operationRecordBefore(const HighsInt operation_type,
                             const HighsInt current_count,
                             const double historical_density);
  void operationRecordAfter(const HighsInt operation_type,
                            const HVector& vector);
  void operationRecordAfter(const HighsInt operation_type,
                            const HighsInt result_count);
  void summaryReport();
  void summaryReportFactor();
  void reportSimplexTimer();
  void reportFactorTimer();
  void updateInvertFormData(const HFactor& factor);
  void reportInvertFormData();

  // Control methods to be moved to HEkkControl
  void dualSteepestEdgeWeightError(const double computed_edge_weight,
                                   const double updated_edge_weight);
  //  bool switchToDevex();

  std::vector<HighsTimerClock> thread_simplex_clocks;
  std::vector<HighsTimerClock> thread_factor_clocks;
  HighsTimerClock* pointer_serial_factor_clocks;

  // Local copies of LP data
  HighsInt numRow;
  HighsInt numCol;
  HighsInt numTot;
  std::string model_name_;
  std::string lp_name_;

  // Local copies of IO data
  HighsLogOptions log_options;

  // Interpreted shortcuts from bit settings in highs_analysis_level
  bool analyse_lp_data;
  bool analyse_simplex_summary_data;
  bool analyse_simplex_runtime_data;
  bool analyse_simplex_time;
  bool analyse_factor_data;
  bool analyse_factor_time;
  bool analyse_simplex_data;

  // Control parameters moving to info
  //  bool allow_dual_steepest_edge_to_devex_switch;
  //  double dual_steepest_edge_weight_log_error_threshold;

  // Local copies of simplex data for reporting
  HighsInt simplex_strategy = 0;
  EdgeWeightMode edge_weight_mode = EdgeWeightMode::kSteepestEdge;
  HighsInt solve_phase = 0;
  HighsInt simplex_iteration_count = 0;
  HighsInt devex_iteration_count = 0;
  HighsInt pivotal_row_index = 0;
  HighsInt leaving_variable = 0;
  HighsInt entering_variable = 0;
  HighsInt rebuild_reason = 0;
  std::string rebuild_reason_string = "";
  double reduced_rhs_value = 0;
  double reduced_cost_value = 0;
  double edge_weight = 0;
  double edge_weight_error = 0;
  double primal_delta = 0;
  double primal_step = 0;
  double dual_step = 0;
  double pivot_value_from_column = 0;
  double pivot_value_from_row = 0;
  double factor_pivot_threshold = 0;
  double numerical_trouble = 0;
  double objective_value = 0;
  HighsInt num_primal_infeasibility = 0;
  HighsInt num_dual_infeasibility = 0;
  double sum_primal_infeasibility = 0;
  double sum_dual_infeasibility = 0;
  // This triple is an original infeasiblility record, so it includes max,
  // but it's only used for reporting
  HighsInt num_dual_phase_1_lp_dual_infeasibility = 0;
  double max_dual_phase_1_lp_dual_infeasibility = 0;
  double sum_dual_phase_1_lp_dual_infeasibility = 0;
  HighsInt num_devex_framework = 0;
  double col_aq_density;
  double row_ep_density;
  double row_ap_density;
  double row_DSE_density;
  double col_steepest_edge_density;
  double col_basic_feasibility_change_density;
  double row_basic_feasibility_change_density;
  double col_BFRT_density;
  double primal_col_density;
  double dual_col_density;
  HighsInt num_costly_DSE_iteration;
  double costly_DSE_measure;

  // Local copies of parallel simplex data for reporting
  HighsInt multi_iteration_count = 0;
  HighsInt multi_chosen = 0;
  HighsInt multi_finished = 0;
  HighsInt min_concurrency = 0;
  HighsInt num_concurrency = 0;
  HighsInt max_concurrency = 0;

  // Unused
  //  HighsInt multi_num = 0; // Useless
  //  double basis_condition = 0; // Maybe useful

  // Records of how pivotal row PRICE was done
  HighsInt num_col_price = 0;
  HighsInt num_row_price = 0;
  HighsInt num_row_price_with_switch = 0;

  HighsValueDistribution before_ftran_upper_sparse_density;
  HighsValueDistribution ftran_upper_sparse_density;
  HighsValueDistribution before_ftran_upper_hyper_density;
  HighsValueDistribution ftran_upper_hyper_density;
  HighsValueDistribution cost_perturbation1_distribution;
  HighsValueDistribution cost_perturbation2_distribution;
  HighsValueDistribution cleanup_dual_change_distribution;
  HighsValueDistribution cleanup_primal_step_distribution;
  HighsValueDistribution cleanup_dual_step_distribution;
  HighsValueDistribution cleanup_primal_change_distribution;

  HighsInt num_primal_cycling_detections = 0;
  HighsInt num_dual_cycling_detections = 0;

  HighsInt num_quad_chuzc = 0;
  HighsInt num_heap_chuzc = 0;
  double sum_quad_chuzc_size = 0;
  double sum_heap_chuzc_size = 0;
  HighsInt max_quad_chuzc_size = 0;
  HighsInt max_heap_chuzc_size = 0;

  HighsInt num_improve_choose_column_row_call = 0;
  HighsInt num_remove_pivot_from_pack = 0;

  HighsInt num_correct_dual_primal_flip = 0;
  double min_correct_dual_primal_flip_dual_infeasibility = kHighsInf;
  double max_correct_dual_primal_flip = 0;
  HighsInt num_correct_dual_cost_shift = 0;
  double max_correct_dual_cost_shift_dual_infeasibility = 0;
  double max_correct_dual_cost_shift = 0;
  HighsInt net_num_single_cost_shift = 0;
  HighsInt num_single_cost_shift = 0;
  double max_single_cost_shift = 0;
  double sum_single_cost_shift = 0;

  // Tolerances for analysis of TRAN stages - could be needed for
  // control if this is ever used again!
  vector<double> original_start_density_tolerance;
  vector<double> new_start_density_tolerance;
  vector<double> historical_density_tolerance;
  vector<double> predicted_density_tolerance;
  vector<TranStageAnalysis> tran_stage;

  std::unique_ptr<std::stringstream> analysis_log;

 private:
  void iterationReport(const bool header);
  void reportAlgorithmPhase(const bool header);
  void reportIterationObjective(const bool header);
  void reportInfeasibility(const bool header);
  void reportThreads(const bool header);
  void reportMulti(const bool header);
  void reportOneDensity(const double density);
  void printOneDensity(const double density);
  void reportDensity(const bool header);
  void reportInvert(const bool header);
  //  void reportCondition(const bool header);
  void reportIterationData(const bool header);
  void reportRunTime(const bool header, const double run_time);
  void reportFreeListSize(const bool header);
  HighsInt intLog10(const double v);
  bool dualAlgorithm();

  //  double AnIterCostlyDseFq;  //!< Frequency of iterations when DSE is costly
  //  double AnIterCostlyDseMeasure;

  HighsInt num_dual_steepest_edge_weight_check = 0;
  HighsInt num_dual_steepest_edge_weight_reject = 0;
  HighsInt num_wrong_low_dual_steepest_edge_weight = 0;
  HighsInt num_wrong_high_dual_steepest_edge_weight = 0;
  double average_frequency_low_dual_steepest_edge_weight = 0;
  double average_frequency_high_dual_steepest_edge_weight = 0;
  double average_log_low_dual_steepest_edge_weight_error = 0;
  double average_log_high_dual_steepest_edge_weight_error = 0;
  double max_average_frequency_low_dual_steepest_edge_weight = 0;
  double max_average_frequency_high_dual_steepest_edge_weight = 0;
  double max_sum_average_frequency_extreme_dual_steepest_edge_weight = 0;
  double max_average_log_low_dual_steepest_edge_weight_error = 0;
  double max_average_log_high_dual_steepest_edge_weight_error = 0;
  double max_sum_average_log_extreme_dual_steepest_edge_weight_error = 0;

  HighsInt num_invert_report_since_last_header = -1;
  HighsInt num_iteration_report_since_last_header = -1;
  double last_user_log_time = -kHighsInf;
  double delta_user_log_time = 1e0;

  double average_concurrency;
  double average_fraction_of_possible_minor_iterations_performed;
  HighsInt sum_multi_chosen = 0;
  HighsInt sum_multi_finished = 0;

  // Analysis of INVERT form
  HighsInt num_invert = 0;
  HighsInt num_kernel = 0;
  HighsInt num_major_kernel = 0;
  double max_kernel_dim = 0;
  double sum_kernel_dim = 0;
  double running_average_kernel_dim = 0;
  double sum_invert_fill_factor = 0;
  double sum_kernel_fill_factor = 0;
  double sum_major_kernel_fill_factor = 0;
  double running_average_invert_fill_factor = 1;
  double running_average_kernel_fill_factor = 1;
  double running_average_major_kernel_fill_factor = 1;

  HighsInt AnIterIt0 = 0;
  HighsInt AnIterPrevIt;

  // Major operation analysis struct
  struct AnIterOpRec {
    double AnIterOpHyperCANCEL;
    double AnIterOpHyperTRAN;
    HighsInt AnIterOpRsDim;
    HighsInt AnIterOpNumCa;
    HighsInt AnIterOpNumHyperOp;
    HighsInt AnIterOpNumHyperRs;
    double AnIterOpSumLog10RsDensity;
    HighsInt AnIterOpRsMxNNZ;
    std::string AnIterOpName;
    HighsValueDistribution AnIterOp_density;
  };
  AnIterOpRec AnIterOp[kNumSimplexNlaOperation];

  struct AnIterTraceRec {
    double AnIterTraceTime;
    double AnIterTraceMulti;
    double AnIterTraceDensity[kNumSimplexNlaOperation];
    double AnIterTraceCostlyDse;
    HighsInt AnIterTraceIter;
    HighsInt AnIterTrace_simplex_strategy;
    HighsInt AnIterTrace_edge_weight_mode;
  };

  HighsInt AnIterTraceNumRec;
  HighsInt AnIterTraceIterDl;
  AnIterTraceRec AnIterTrace[1 + kAnIterTraceMaxNumRec + 1];

  HighsInt AnIterNumInvert[kRebuildReasonCount];
  HighsInt AnIterNumEdWtIt[(HighsInt)EdgeWeightMode::kCount];

  HighsValueDistribution primal_step_distribution;
  HighsValueDistribution dual_step_distribution;
  HighsValueDistribution simplex_pivot_distribution;
  HighsValueDistribution numerical_trouble_distribution;
  HighsValueDistribution factor_pivot_threshold_distribution;
  HighsValueDistribution edge_weight_error_distribution;
};

#endif /* SIMPLEX_HIGHSSIMPLEXANALYSIS_H_ */
