import numpy as np
from pandas import DataFrame


from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator


from catboost import (
    CatBoostClassifier,
)


def test_sklearn_meta_algo():
    X_train = DataFrame(
        data=np.random.randint(0, 100, size=(100, 5)),
        columns=['feature{}'.format(i) for i in range(5)]
    )
    y_train = np.random.randint(0, 2, size=100)

    model = CatBoostClassifier()
    model.fit(X_train, y_train)

    cc_model = CalibratedClassifierCV(FrozenEstimator(model), method='isotonic')
    model = cc_model.fit(X_train, y_train)
