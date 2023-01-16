from __future__ import print_function

import os
import time

from .. import CatBoostError
from ..eval.log_config import get_eval_logger
from ..eval.utils import make_dirs_if_not_exists
from ..eval.evaluation_result import CaseEvaluationResult
from ._fold_model import FoldModel


class FoldModelsHandler(object):
    """
    Class that is responsible for learning models and computing their metrics
    """
    """
       All models are placed to the default directory "models".
       """
    __MODEL_DIR = 'models'

    @staticmethod
    def _remove_model_dir():
        try:
            if os.path.exists(FoldModelsHandler.__MODEL_DIR):
                os.rmdir(FoldModelsHandler.__MODEL_DIR)
        except OSError as err:
            get_eval_logger().warning(str(err))

    def __init__(self, metrics, cases, thread_count, eval_step, remove_models):
        """
        Args:
            :param remove_models: Set true if you want models to be removed after applying them.

        """
        self._cases = cases
        self._metrics = metrics

        self._case_results = dict()
        for case in self._cases:
            self._case_results[case] = dict()

        self._thread_count = thread_count
        self._eval_step = eval_step
        self._flag_remove_models = remove_models
        self._metric_descriptions = None

    def _init_case_results(self, metric_descriptions):
        self._metric_descriptions = metric_descriptions
        for case in self._cases:
            case_result = self._case_results[case]
            for metric_description in self._metric_descriptions:
                case_result[metric_description] = CaseEvaluationResult(case,
                                                                       metric_description,
                                                                       eval_step=self._eval_step)

    def _compute_metrics(self, metrics, grouped_by_case_models, learn_folds, skipped_folds, rest_folds):
        metric_calcers = {}

        for case, case_models in grouped_by_case_models.items():
            metric_calcers[case] = list()
            for case_model in case_models:

                metric_calcer = case_model.create_metrics_calcer(metrics,
                                                                 eval_step=self._eval_step,
                                                                 thread_count=self._thread_count)
                metric_calcers[case].append(metric_calcer)

                if self._metric_descriptions is None:
                    self._init_case_results(metric_calcer.metric_descriptions())
                elif self._metric_descriptions != metric_calcer.metric_descriptions():
                    raise CatBoostError("Error: metric names should be consistent")

        for file_num, fold_file in enumerate(learn_folds + skipped_folds + rest_folds):
            pool = FoldModelsHandler._create_pool(fold_file, self._thread_count)

            for case, case_models in grouped_by_case_models.items():
                calcers = metric_calcers[case]

                for model_num, model in enumerate(case_models):
                    if file_num != model_num:
                        calcers[model_num].add(pool)

        for case, case_models in grouped_by_case_models.items():
            calcers = metric_calcers[case]
            case_results = self._case_results[case]
            for calcer, model in zip(calcers, case_models):
                scores = calcer.eval_metrics()
                for metric in self._metric_descriptions:
                    case_results[metric]._add(model, scores.get_result(metric))

    @staticmethod
    def _fit_model(pool, case, fold_id, model_path):
        from .. import CatBoost
        # Learn model
        make_dirs_if_not_exists(FoldModelsHandler.__MODEL_DIR)

        feature_count = pool.num_col()
        if "ignored_features" in case.get_params():
            ignored_features = case.get_params()["ignored_features"]
            if len(ignored_features) and max(ignored_features) >= feature_count:
                raise CatBoostError("Error: input parameter contains feature indices wich are not available in pool: "
                                    "{}\n "
                                    "Check eval_feature set and ignored features options".format(ignored_features))
        get_eval_logger().debug('Learn model {} on fold #{}'.format(str(case), fold_id))
        cur_time = time.time()
        instance = CatBoost(params=case.get_params())
        instance.fit(pool)
        instance.save_model(fname=model_path)

        get_eval_logger().debug('Operation was done in {} seconds'.format(time.time() - cur_time))
        return FoldModel(case, model_path, fold_id)

    def _fit_models(self, learn_files, fold_id_bias):
        """
        Train models for each algorithm and learn dataset(folds). Than return them.

        Args:
            :param learn_files: Entities of FoldStorage for learning models.
            :return: Dictionary of models where the key is case and the value is models on learn folds
        """
        make_dirs_if_not_exists(FoldModelsHandler.__MODEL_DIR)

        models = {}
        for case in self._cases:
            models[case] = list()

        for file_num, learn_file in enumerate(learn_files):
            pool = FoldModelsHandler._create_pool(learn_file, self._thread_count)
            fold_id = fold_id_bias + file_num

            for case in self._cases:
                model_path = os.path.join(FoldModelsHandler.__MODEL_DIR,
                                          FoldModelsHandler._create_model_name(case, fold_id))
                get_eval_logger().debug("For model {} on fold #{} path is {}".format(str(case), fold_id, model_path))
                fold_model = self._fit_model(pool, case, fold_id, model_path)
                get_eval_logger().info("Model {} on fold #{} was fitted".format(str(case), fold_id))
                models[case].append(fold_model)

        return models

    def proceed(self, splitter, fold_size, folds_count, fold_offset):
        """
        Run all processes to gain stats. It applies algorithms to fold files that gains from learning. It keeps
        stats inside models and models are stored in DataFrame. Columns are matched to the different algos and rows to
        the folds.

        Args:
            :param splitter: Splitter entity.
            :param fold_size: The size of fold.
            :param folds_count: Count of golds.
            :param fold_offset: The offset (count of folds that we want to skip).
            :return: return dict: keys metric to CaseEvaluationResult

        """
        try:
            folds_sets = splitter.create_fold_sets(fold_size, folds_count)
            fold_groups_files = splitter.fold_groups_files_generator(folds_sets,
                                                                     fold_offset)
            fold_id_bias = fold_offset

            for learn_folds, skipped_folds, rest_folds in fold_groups_files:
                if len(learn_folds) == 0:
                    continue
                list_models = []
                try:
                    permutation_models = self._fit_models(learn_folds, fold_id_bias)
                    for case, case_models in permutation_models.items():
                        list_models += case_models

                    learn_folds_count = len(learn_folds)
                    get_eval_logger().info("Start metric computation for folds [{}, {})"
                                           .format(fold_id_bias, fold_id_bias + learn_folds_count))
                    self._compute_metrics(self._metrics,
                                          permutation_models,
                                          learn_folds, skipped_folds, rest_folds)
                    get_eval_logger().info("Computation of metrics for  folds [{}, {}) is completed"
                                           .format(fold_id_bias, fold_id_bias + learn_folds_count))
                    fold_id_bias += learn_folds_count
                finally:
                    # Do it each step because don't want to occupy a lot of memory
                    splitter.clean_folds()
                    self._remove_models(list_models)

        finally:
            # Also sometimes we need to remove models and always need to try to remove folds directory.
            self._clean(splitter)

        return self._case_results

    def _clean(self, splitter):
        if self._flag_remove_models:
            self._remove_model_dir()
        splitter.clean()

    @staticmethod
    def _remove_models(list_models):
        get_eval_logger().debug('Remove models {}'.format(str(list_models)))
        for model in list_models:
            model.delete()

    @staticmethod
    def _create_model_name(model_case, fold):
        import uuid
        id_str = str(uuid.uuid1()).replace("-", "_")
        model_name = "model_{_name}_fold_{_fold}_{_uuid}.bin".format(_name=model_case.get_label(),
                                                                     _fold=fold,
                                                                     _uuid=id_str)
        return model_name

    @staticmethod
    def _create_pool(fold_file, thread_count=-1):
        from .. import Pool
        data_pool = Pool(fold_file.path(),
                         column_description=fold_file.column_description(),
                         delimiter=fold_file.get_separator(),
                         thread_count=thread_count)
        return data_pool
