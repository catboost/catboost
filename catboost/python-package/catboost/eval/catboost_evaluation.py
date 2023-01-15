from __future__ import print_function

import os
from copy import copy
from enum import Enum

from catboost import CatBoostError, CatBoost
from .evaluation_result import EvaluationResults, MetricEvaluationResult
from ._fold_models_handler import FoldModelsHandler
from ._readers import _SimpleStreamingFileReader
from ._splitter import _Splitter
from .execution_case import ExecutionCase
from .factor_utils import LabelMode, FactorUtils


class EvalType(Enum):
    """
        Type of feature evaluation:
            All: All factors presented
            SeqRem:  Each factor while other presented
            SeqAdd:  Each factor while other removed
            SeqAddAndAll:  SeqAdd + All
    """
    All = 'All'
    SeqRem = 'SeqRem'
    SeqAdd = 'SeqAdd'
    SeqAddAndAll = 'SeqAddAndAll'


class CatboostEvaluation(object):

    def __init__(self,
                 path_to_dataset,
                 fold_size,
                 fold_count,
                 column_description,
                 fold_offset=0,
                 group_column=None,
                 working_dir=None,
                 remove_models=True,
                 delimiter='\t',
                 has_header=False,
                 partition_random_seed=0,
                 min_fold_count=1):
        """
        Args:
            :param path_to_dataset: (str) Path to the dataset to be used for evaluation.
            :param fold_size: (int) Size of the folds in cross-validation.
            :param fold_count: (int) Number of times we get a new fold, learn a model and check results as if
            there wouldn't be any offset. If there'are some offset it means that the real count of folds will
            be smaller.
            :param column_description: (str) Path to the file where column description is placed.
            :param fold_offset: (int) Number of folds we skip before begin to make cross-validation.
            :param group_column: (int) GroupId column index in the dataset file.
            'None' value means absence of grouping information in the dataset (it's the default).
            :param working_dir: Working dir for temporary files
            :param remove_models: (bool) Set true if you want models to be removed after applying them.
            :param delimiter: (str) Field delimiter used in dataset files.
            :param has_header: (bool) Set true if you want to skip first line in dataset files.
            :param partition_random_seed: (int) The seed for random value generator used for getting permutations for
             cross-validation.
            :param min_fold_count: (int) Minimun amount of folds dataset can be cut to.
        """
        import os.path

        self._current_dir = os.getcwd()
        self._path_to_dataset = os.path.join(self._current_dir, path_to_dataset)
        self._column_description = os.path.join(self._current_dir,
                                                column_description) if column_description is not None else None
        self._fold_offset = fold_offset
        self._fold_count = fold_count
        self._fold_size = fold_size
        self._delimiter = delimiter
        self._has_header = has_header
        self._seed = partition_random_seed
        self._min_fold_count = int(min_fold_count)
        self._remove_models = remove_models

        if group_column is not None:
            self._group_feature_num = int(group_column)
        else:
            self._group_feature_num = group_column

        if working_dir is None:
            import tempfile
            self._working_dir = tempfile.mkdtemp()
        else:
            self._working_dir = working_dir

    def __go_to_working_dir(self):
        current = os.getcwd()
        os.chdir(self._working_dir)
        return current

    @staticmethod
    def _create_eval_feature_cases(params, features_to_eval, eval_type, label_mode):
        if len(features_to_eval) == 0:
            raise CatBoostError("Provide at least one feature to evaluation")

        # baseline
        test_cases = list()
        baseline_case = ExecutionCase(params,
                                      ignored_features=list(features_to_eval),
                                      label=FactorUtils.create_label(features_to_eval,
                                                                     features_to_eval,
                                                                     label_mode=label_mode)
                                      )
        # test
        if eval_type == EvalType.All or eval_type == EvalType.SeqAddAndAll or len(features_to_eval) == 1:
            test_cases.append(ExecutionCase(params,
                                            ignored_features=[],
                                            label=FactorUtils.create_label(features_to_eval,
                                                                           [],
                                                                           label_mode=label_mode)
                                            ))
        elif eval_type == EvalType.SeqRem:
            for feature_num in features_to_eval:
                test_cases.append(ExecutionCase(params,
                                                ignored_features=[feature_num],
                                                label=FactorUtils.create_label(features_to_eval,
                                                                               [feature_num],
                                                                               label_mode=label_mode)
                                                ))
        elif eval_type == EvalType.SeqAdd or eval_type == EvalType.SeqAddAndAll:
            for feature_num in features_to_eval:
                cur_features = copy(features_to_eval)
                cur_features.remove(feature_num)
                test_cases.append(ExecutionCase(params,
                                                label=FactorUtils.create_label(features_to_eval,
                                                                               cur_features,
                                                                               label_mode=label_mode),
                                                ignored_features=list(cur_features)))
        elif eval_type != EvalType.All:
            raise AttributeError("Don't support {} mode.", eval_type.value)
        return baseline_case, test_cases

    @staticmethod
    def _create_evaluation_results(by_case_results):
        group_by_metric = dict()

        for (case, case_result) in by_case_results.items():
            for (metric, evaluation_result) in case_result.items():
                if metric not in group_by_metric:
                    group_by_metric[metric] = list()
                group_by_metric[metric].append(evaluation_result)

        results = list()
        for (metric, metric_results) in group_by_metric.items():
            results.append(MetricEvaluationResult(metric_results))
        return EvaluationResults(results)

    def get_working_dir(self):
        return self._working_dir

    def _calculate_result_metrics(self, cases, metrics, thread_count=-1, evaluation_step=1):
        """
        This method calculate metrics and return them.

        Args:
            :param cases: List of the ExecutionCases you want to evaluate
            :param metrics: List of the metrics to be computed
            :param thread_count: Count of threads to use.
            :param: evaluation_step: Step to evaluate metrics
            :return: instance of EvaluationResult
        """
        cases_set = set(cases)
        if len(cases_set) != len(cases):
            raise CatBoostError("Found duplicate cases in " + cases)
        current_wd = self.__go_to_working_dir()
        try:
            if self._fold_count <= self._fold_offset:
                error_msg = 'Count of folds(folds_count - offset) need to be at least one: offset {}, folds_count {}.'
                raise AttributeError(error_msg.format(self._fold_offset,
                                                      self._fold_count))

            handler = FoldModelsHandler(cases=cases,
                                        metrics=metrics,
                                        eval_step=evaluation_step,
                                        thread_count=thread_count,
                                        remove_models=self._remove_models)

            reader = _SimpleStreamingFileReader(self._path_to_dataset,
                                                sep=self._delimiter,
                                                has_header=self._has_header,
                                                group_feature_num=self._group_feature_num)
            splitter = _Splitter(reader,
                                 self._column_description,
                                 seed=self._seed,
                                 min_folds_count=self._min_fold_count)

            result = handler.proceed(splitter=splitter,
                                     fold_size=self._fold_size,
                                     folds_count=self._fold_count,
                                     fold_offset=self._fold_offset)

            return self._create_evaluation_results(result)
        finally:
            os.chdir(current_wd)

    def eval_features(self,
                      learn_config,
                      features_to_eval,
                      loss_function=None,
                      eval_type=EvalType.SeqAdd,
                      eval_metrics=None,
                      thread_count=-1,
                      eval_step=None,
                      label_mode=LabelMode.AddFeature):
        """ Evaluate features.
            Args:
            learn_config: dict with params or instance of CatBoost. In second case instance params will be used
            features_to_eval: list of indices of features to evaluate
            loss_function: one of CatBoost loss functions, get it from learn_config if not specified
            eval_type: Type of feature evaluate (All, SeqAdd, SeqRem)
            eval_metrics: Additional metrics to calculate
            thread_count: thread_count to use. If not none will override learn_config values
            Returns
            -------
            result : Instance of EvaluationResult class
        """
        features_to_eval = set(features_to_eval)
        if eval_metrics is None:
            eval_metrics = []
        eval_metrics = eval_metrics if isinstance(eval_metrics, list) else [eval_metrics]
        if isinstance(learn_config, CatBoost):
            params = learn_config.get_params()
        else:
            params = dict(learn_config)

        if loss_function is not None:
            if "loss_function" in params and params["loss_function"] != loss_function:
                raise CatBoostError("Loss function in params {} should be equal to feature evaluation objective "
                                    "function {}".format(params["loss_function"], loss_function))
        else:
            if "loss_function" not in params:
                raise CatBoostError("Provide loss function in params or as option to eval_features method")

        if thread_count is not None and thread_count != -1:
            params["thread_count"] = thread_count

        if eval_step is None:
            eval_step = 1

        if loss_function is not None:
            params["loss_function"] = loss_function
        else:
            loss_function = params["loss_function"]

        if params["loss_function"] == "PairLogit":
            raise CatBoostError("Pair classification is not supported")

        baseline_case, test_cases = self._create_eval_feature_cases(params,
                                                                    features_to_eval,
                                                                    eval_type=eval_type,
                                                                    label_mode=label_mode)
        if loss_function not in eval_metrics:
            eval_metrics.append(loss_function)

        return self.eval_cases(baseline_case=baseline_case,
                               compare_cases=test_cases,
                               eval_metrics=eval_metrics,
                               thread_count=thread_count,
                               eval_step=eval_step)

    def eval_cases(self,
                   baseline_case,
                   compare_cases,
                   eval_metrics,
                   thread_count=-1,
                   eval_step=1):
        """More flexible evaluation of any cases.
            Args:
            baseline_case: Execution case used for baseline
            compare_cases: List of cases to compare
            eval_metrics: Metrics to calculate
            thread_count: thread_count to use.  Will override one in cases
            Returns
            -------
            result : Instance of EvaluationResult class
        """
        if not isinstance(compare_cases, list):
            compare_cases = [compare_cases]

        cases = [baseline_case]
        cases += compare_cases

        for case in cases:
            case._set_thread_count(thread_count)

        metric_result = self._calculate_result_metrics(cases,
                                                       eval_metrics,
                                                       thread_count=thread_count,
                                                       evaluation_step=eval_step)
        metric_result.set_baseline_case(baseline_case)
        return metric_result
