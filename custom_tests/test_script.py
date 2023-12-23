from catboost import CatBoostClassifier
from catboost import Pool
import math
import numpy as np

class LoglossMetric(object):
    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]

        error_sum = 0.0
        weight_sum = 0.0

        for i in range(len(approx)):
            e = np.exp(approx[i])
            p = e / (1 + e)
            w = 1.0 if weight is None else weight[i]
            weight_sum += w
            error_sum += -w * (target[i] * np.log(p) + (1 - target[i]) * np.log(1 - p))

        return error_sum, weight_sum


TRAIN_FILE = '/home/catboost/catboost/pytest/data/adult/train_small'
TEST_FILE = '/home/catboost/catboost/pytest/data/adult/test_small'
CD_FILE = '/home/catboost/catboost/pytest/data/adult/train.cd'
CD_FILE_NO_TARGET = '/home/catboost/catboost/pytest/data/adult/test_small/train_no_target.cd'

train_pool = Pool(data=TRAIN_FILE, column_description=CD_FILE)
test_pool = Pool(data=TEST_FILE, column_description=CD_FILE)

model = CatBoostClassifier(iterations=5, learning_rate=0.03, use_best_model=True,
                            loss_function="Logloss", eval_metric=LoglossMetric(),
                            # Leaf estimation method and gradient iteration are set to match
                            # defaults for Logloss.
                            leaf_estimation_method="Newton", leaf_estimation_iterations=1, task_type='GPU', devices='0',
                            metric_period=1
)

model.fit(train_pool, eval_set=test_pool)
pred1 = model.predict(test_pool, prediction_type='RawFormulaVal')

model2 = CatBoostClassifier(
    iterations=5,
    learning_rate=0.03,
    use_best_model=True,
    loss_function="Logloss",
    task_type='GPU',
    leaf_estimation_method="Newton",
    leaf_estimation_iterations=1
)

model2.fit(train_pool, eval_set=test_pool)
pred2 = model2.predict(test_pool, prediction_type='RawFormulaVal')

for p1, p2 in zip(pred1, pred2):
    assert abs(p1 - p2) < 0.0001
print("max diff: {}".format(np.max(np.abs(pred2 - pred1))))