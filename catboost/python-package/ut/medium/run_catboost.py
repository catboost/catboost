import six

from catboost import CatBoostClassifier, cv
from catboost.datasets import titanic, adult
import catboost
import numpy as np


def test_titanic():
    train_df = titanic()[0].fillna(-999)
    X, y = train_df.drop('Survived', axis=1), train_df.Survived
    categorical_features_indices = np.where(X.dtypes != np.float)[0]

    model = CatBoostClassifier(iterations=5)
    model.fit(X, y, cat_features=categorical_features_indices)
    preds = model.predict(X)


def test_adult():
    train, test = adult()

    # CatBoost doesn't support pandas.DataFrame NaNs out of the box for categorical features,
    # so we'll replace them manually with some special string (we'll use "nan")
    #
    # seed issue #571 on GitHub or issue MLTOOLS-2785 in internal tracker.
    #
    # oh, and don't forget to replace missing values with string "nan" when you are going to apply
    # the model!
    for dataset in (train, test, ):
        for name in (name for name, dtype in six.iteritems(dict(dataset.dtypes)) if dtype == np.object):
            dataset[name].fillna('nan', inplace=True)

    X_train, y_train = train.drop('income', axis=1), train.income
    X_test, y_test = test.drop('income', axis=1), test.income
    model = CatBoostClassifier(iterations=5, loss_function='CrossEntropy', class_names=['<=50K', '>50K'])
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test, ),
        cat_features=np.where(X_train.dtypes != np.float)[0],)

    predictions = model.predict(X_test)

if __name__ == '__main__':
    test_titanic()
    test_adult()
