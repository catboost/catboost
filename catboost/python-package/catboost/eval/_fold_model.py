"""
This class provides the abstraction of model. I.e. it is an object relevant to particular learn fold and particular
algorithm with parameters.
"""

import os
from .. import CatBoostError, CatBoost


class FoldModel:

    def __init__(self, case, model_path, model_id):
        self._id = model_id
        self._case = case
        self._model_path = model_path

    def __repr__(self):
        return 'Model_{}_id_{}'.format(str(self._case), self._id)

    def create_metrics_calcer(self, metrics, thread_count, eval_step=1):
        if not os.path.exists(self._model_path):
            raise CatBoostError("Model was deleted. Can't create calcer now")
        model = CatBoost()
        model.load_model(self._model_path)
        return model.create_metric_calcer(metrics, thread_count=thread_count, eval_period=eval_step)

    def get_case(self):
        return self._case

    def get_fold_id(self):
        return self._id

    def delete(self):
        if os.path.exists(self._model_path):
            os.remove(self._model_path)
