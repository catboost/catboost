import catboost
from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor, cv, Pool, train, utils
from catboost.datasets import titanic, amazon
from catboost.utils import create_cd
import numpy as np

def test():
    train_df = titanic()[0].fillna(-999)
    X, y = train_df.drop('Survived', axis=1), train_df.Survived
    categorical_features_indices = np.where(X.dtypes != np.float)[0]

    model = CatBoostClassifier(iterations=5)
    model.fit(X, y, cat_features=categorical_features_indices)
    preds = model.predict(X)
    pred_single = model.predict(X[0])

if __name__ == '__main__':
    test()
