import numpy as np
import pandas as pd
from enum import Enum

from catboost import CatboostError
from catboost.core import metric_description_or_str_to_str
from catboost.core import compute_wx_test


def calc_wilcoxon_test(baseline, test):
    return compute_wx_test(baseline, test)["pvalue"]


class ScoreType(Enum):
    Abs = "AbsoluteDiff"
    Rel = "RelativeDiff"


class ScoreConfig:
    """
        Config to present human-friendly evaluation results.
    """

    def __init__(self,
                 score_type=ScoreType.Rel,
                 multiplier=100,
                 score_level=0.01,
                 interval_level=0.01,
                 overfit_iterations_info=True):
        """
        :param multiplier: multiplier to print score
        :param score_type: type of score. For abs difference score will be (baseline - test).mean(), for relative it's ((baseline - test) / baseline).mean()
        :param score_level: WX-test level. Will be used to make if tested case significantly better or worse
        :param interval_level: level to compute score confidence interval
        :param overfit_iterations_info: if information about oberfit iterations should be prested
        """
        self.type = score_type
        self.multiplier = multiplier
        self.score_level = score_level
        self.interval_level = interval_level
        self.overfit_overfit_iterations_info = overfit_iterations_info

    @staticmethod
    def abs_score(level=0.01):
        return ScoreConfig(score_type=ScoreType.Abs,
                           multiplier=1,
                           score_level=level)

    @staticmethod
    def rel_score(level=0.01):
        return ScoreConfig(score_type=ScoreType.Rel,
                           multiplier=100,
                           score_level=level)


def calc_bootstrap_ci_for_mean(samples, level=0.05, tries=999):
    """
    Count confidence intervals for difference each two samples.

    Args:
        :param samples: samples
        :param level: (float) Level for the confidence interval.
        :param tries: bootstrap samples to use
        :return: (left, right) border of confidence interval

    """

    if not (samples == 0).all():
        samples = np.array(samples)
        means = []
        for _ in range(0, tries):
            resample = np.random.choice(samples, len(samples))
            means.append(np.mean(resample))
        means = sorted(means)
        left = means[int(tries * (level / 2))]
        right = means[int(tries * (1.0 - level / 2))]
        return left, right
    else:
        return 0, 0


class CaseEvaluationResult:
    """
        CaseEvaluationResults stores aggregated statistics for once EvaluationCase and one metric.
    """

    def __init__(self, case, metric_description, eval_step):
        self._case = case
        self._metric_description = metric_description

        self._fold_metric = pd.Series()
        self._fold_metric_iteration = pd.Series()
        self._fold_curves = dict()

        self._eval_step = eval_step

    def _add(self, model, learning_curve):
        if model.get_case() != self._case:
            raise CatboostError("Model case should be equal to result case")

        fold_id = model.get_fold_id()

        self._fold_curves[fold_id] = learning_curve
        score = max(learning_curve) if self._metric_description.is_max_optimal() else min(learning_curve)
        position = np.argmax(learning_curve) if self._metric_description.is_max_optimal() else np.argmin(
            learning_curve)

        self._fold_metric.at[fold_id] = score
        self._fold_metric_iteration.at[fold_id] = position

    def __eq__(self, other):
        return (np.all(self._fold_metric == other._fold_metric)
                and np.all(self._fold_metric_iteration == other._fold_metric_iteration)
                and self._fold_curves == other._fold_curves)

    def get_case(self):
        """
            ExecutionCases for this result
        """
        return self._case

    def get_fold_ids(self):
        """
        :return: FoldsIds for which this caseResult was calculatee
        """
        return self._fold_curves.keys()

    def get_best_metric_for_fold(self, fold):
        """
        :param fold: id of fold to get result
        :return: best metric value, best metric iteration
        """
        return self._fold_metric[fold], self._fold_metric_iteration[fold]

    def get_best_iterations(self):
        """
        :return: pandas Series with best iterations on all folds
        """
        return self._fold_metric_iteration

    def get_best_metrics(self):
        """
        :return: pandas series with best metric values
        """
        return self._fold_metric

    def get_fold_curve(self, fold):
        """
        :param fold:
        :return: fold learning curve (test scores on every eval_period iteration)
        """
        return self._fold_curves[fold]

    def get_metric_description(self):
        """

        :return: Metric used to build this CaseEvaluationResult
        """
        return self._metric_description

    def get_eval_step(self):
        """
        :return: step which was used for metric computations
        """
        return self._eval_step

    def count_under_and_over_fits(self, overfit_border=0.15, underfit_border=0.95):
        """

        :param overfit_border: min fraction of iterations untill overffiting stars one excpects all models to have
        :param underfit_border: border, after which there should be no best_metric_scores
        :return: #models with best_metric > underfit_border * iter_count, #models, with best_metric > overfit_border
        """
        count_overfitting = 0
        count_underfitting = 0

        for fold_id, fold_curve in self._fold_curves.items():
            best_score_position = self._fold_metric_iteration[fold_id]
            best_model_size_fraction = best_score_position * 1.0 / len(fold_curve)
            if best_model_size_fraction > overfit_border:
                count_underfitting += 1
            elif best_model_size_fraction < underfit_border:
                count_overfitting += 1
        return count_overfitting, count_underfitting

    def estimate_fit_quality(self):
        """
        :return: Simple sanity check that all models overfitt and not too fast
        """
        count_overfitting, count_underfitting = self.count_under_and_over_fits()
        if count_overfitting > count_underfitting:
            return "Overfitting"
        if count_underfitting > count_overfitting:
            return "Underfitting"
        return "Good"

    def create_learning_curves_plot(self, offset=None):
        """
        :param offset: First iteration to plot
        :return: plotly Figure with learning curves for each fold
        """
        import plotly.graph_objs as go

        traces = []

        for fold in self.get_fold_ids():
            scores_curve = self.get_fold_curve(fold)
            if offset is not None:
                first_idx = offset
            else:
                first_idx = int(len(scores_curve) * 0.1)

            traces.append(go.Scatter(x=[i * int(self._eval_step) for i in range(first_idx, len(scores_curve))],
                                     y=scores_curve[first_idx:],
                                     mode='lines',
                                     name='Fold #{}'.format(fold)))

        layout = go.Layout(
            title='Learning curves for case {}'.format(self._case),
            hovermode='closest',
            xaxis=dict(
                title='Iteration',
                ticklen=5,
                zeroline=False,
                gridwidth=2,
            ),
            yaxis=dict(
                title='Metric',
                ticklen=5,
                gridwidth=2,
            ),
            showlegend=True
        )
        fig = go.Figure(data=traces, layout=layout)
        return fig


class MetricEvaluationResult:
    """
        Evaluation result for one metric.
        Stores all ExecutionCases with specified metric scores
        Computes human-friendly tables with results and some plots
    """

    def __init__(self, case_results):
        if len(case_results) <= 1:
            raise CatboostError("Need at least 2 case results, got {} ".format(len(case_results)))

        self._case_results = dict()
        self._case_comparisons = dict()
        self._cases = [case_result.get_case() for case_result in case_results]

        for case_result in case_results:
            case = case_result.get_case()
            self._case_results[case] = case_result

        self._metric_description = case_results[0].get_metric_description()
        self._baseline_case = case_results[0].get_case()

        self._score_config = ScoreConfig()

        for (case, case_result) in self._case_results.items():
            if case_result.get_metric_description() != self._metric_description:
                raise CatboostError("Metric names should be equal for all case results")

            if case_result.get_fold_ids() != self.get_fold_ids():
                raise CatboostError("Case results should be computed on the same folds")

            if case_result.get_eval_step() != self.get_eval_step():
                raise CatboostError("Eval steps should be equal for different cases")

    def __clear_comparisons(self):
        self._case_comparisons = dict()

    def _change_score_config(self, config):
        if config is not None:
            if isinstance(config, ScoreType):
                if config == ScoreType.Abs:
                    config = ScoreConfig.abs_score()
                elif config == ScoreType.Rel:
                    config = ScoreConfig.rel_score()
                else:
                    raise CatboostError("Unknown scoreType {}".format(config))
            if self._score_config != config:
                self._score_config = config
                self.__clear_comparisons()

    def _compute_case_result_table(self, baseline_case):
        result = pd.DataFrame()
        baseline_scores = self._case_results[baseline_case].get_best_metrics()
        baseline_iters = self._case_results[baseline_case].get_best_iterations()

        for (case, case_result) in self._case_results.items():
            if case != baseline_case:
                test_scores = case_result.get_best_metrics()
                pvalue = calc_wilcoxon_test(baseline_scores, test_scores)

                diff = (baseline_scores - test_scores)
                if self._score_config.type == ScoreType.Rel:
                    diff = diff / baseline_scores.abs()
                if self._metric_description.is_max_optimal():
                    diff = -diff
                mean_diff = diff.mean()

                left_quantile, right_quantile = calc_bootstrap_ci_for_mean(diff,
                                                                           self._score_config.interval_level)

                result.at[case, "PValue"] = pvalue
                result.at[case, "Score"] = mean_diff * self._score_config.multiplier

                left_quantile_title = "Quantile {}".format(self._score_config.score_level / 2)
                right_quantile_title = "Quantile {}".format(1.0 - self._score_config.score_level / 2)

                result.at[case, left_quantile_title] = left_quantile * self._score_config.multiplier
                result.at[case, right_quantile_title] = right_quantile * self._score_config.multiplier

                decision = "UNKNOWN"
                if pvalue < self._score_config.score_level:
                    if mean_diff > 0:
                        decision = "GOOD"
                    elif mean_diff < 0:
                        decision = "BAD"
                result.at[case, "Decision"] = decision

                if self._score_config.overfit_overfit_iterations_info:
                    test_iters = case_result.get_best_iterations()
                    pvalue = calc_wilcoxon_test(baseline_iters, test_iters)

                    result.at[case, "Overfit iter diff"] = (test_iters - baseline_iters).mean()
                    result.at[case, "Overfit iter pValue"] = pvalue

        return result.sort_values(by=["Score"], ascending=self._metric_description.is_max_optimal())

    def get_baseline_case(self):
        """

        :return: ExecutionCases used as baseline (with everything else is compared)
        """
        return self._baseline_case

    def get_cases(self):
        """

        :return: Cases which are compared
        """
        return self._cases

    def get_metric_description(self):
        """

        :return: Metric for which results were calculated
        """
        return self._metric_description

    def get_baseline_comparison(self, score_config=None):
        """
        Method to get human-friendly table with model comparisons.
        Returns baseline vs all other computed cases result
        :param score_config: Config to present human-friendly score, optional. Instance of ScoreConfig
        :return: pandas DataFrame. Each row is related to one ExecutionCase.
        Each row describes how better (or worse) this case compared to baseline.
        """
        case = self._baseline_case
        return self.get_case_comparison(case, score_config)

    def get_case_comparison(self, case, score_config=None):
        """
        Same as previous method, but with other non-baseline case specified as baseline
        :param case: use specified case as baseline
        :param score_config:
        :return:
        """
        self._change_score_config(score_config)
        if case not in self._case_comparisons:
            self._case_comparisons[case] = self._compute_case_result_table(case)
        return self._case_comparisons[case]

    def change_baseline_case(self, case):
        """
        :param case: new baseline case
        :return:
        """
        if case not in self._case_results:
            raise CatboostError("Case {} is unknown. Can't use it as baseline".format(case))
        self._baseline_case = case

    def get_case_result(self, case):
        """
        :param case:
        :return: CaseEvaluationResult. Scores and other information about single execution case
        """
        return self._case_results[case]

    def get_fold_ids(self):
        """

        :return: Folds ids which we used for computing this evaluation result
        """
        return self._case_results[self._baseline_case].get_fold_ids()

    def get_eval_step(self):
        return self._case_results[self._baseline_case].get_eval_step()

    def create_fold_learning_curves(self, fold, offset=None):
        """

        :param fold: FoldId to plot
        :param offset: first iteration to plot
        :return: plotly figure for all cases on specified fold
        """
        import plotly.graph_objs as go
        traces = []
        for case in self.get_cases():
            case_result = self.get_case_result(case)
            scores_curve = case_result.get_fold_curve(fold)
            if offset is not None:
                first_idx = offset
            else:
                first_idx = int(len(scores_curve) * 0.1)

            traces.append(
                go.Scatter(x=[i * int(case_result.get_eval_step()) for i in range(first_idx, len(scores_curve))],
                           y=scores_curve[first_idx:],
                           mode='lines',
                           name='Case {}'.format(case)))

        layout = go.Layout(
            title='Learning curves for metric {} on fold #{}'.format(self._metric_description, fold),
            hovermode='closest',
            xaxis=dict(
                title='Iteration',
                ticklen=5,
                zeroline=False,
                gridwidth=2,
            ),
            yaxis=dict(
                title='Metric',
                ticklen=5,
                gridwidth=2,
            ),
            showlegend=True
        )
        fig = go.Figure(data=traces, layout=layout)
        return fig

    def __eq__(self, other):
        return (self._case_results == other._case_results
                and self._case_comparisons == other._case_comparisons
                and self._cases == other._cases)


class EvaluationResults:

    def __init__(self, metric_results):
        if len(metric_results) < 1:
            raise CatboostError("Need at least one result")

        self._results = dict()
        self._metrics = dict()
        self._cases = None

        for result in metric_results:
            metric_description = result.get_metric_description()
            if metric_description in self._results:
                raise CatboostError("Duplicate metric {}".format(metric_description))
            if self._cases is None:
                self._cases = result.get_cases()
            key = metric_description_or_str_to_str(metric_description)
            self._results[key] = result
            self._metrics[key] = metric_description

    def get_metric_results(self, metric):
        """

        :param metric:
        :return: MetricEvaluationResult for specified metric
        """
        return self._results[metric_description_or_str_to_str(metric)]

    def get_metrics(self):
        """
        Returns
        -------
            Metric names which were computed
        """
        return self._metrics

    def get_results(self):
        """
        Returns
        -------
            Results are map from metric names to computed results (instance of MetricEvaluationResult) on this fold
        """
        return self._results

    def set_baseline_case(self, case):
        """
            Could be used to change baseline cases for already computed results
        """
        for (metric, metric_result) in self._results.items():
            metric_result.change_baseline_case(case)
