import os
import six

from catboost import CatBoostClassifier, Pool


def data_file(*path):
    return os.path.join(os.getcwd(), "data", *path)


def test_adult():
    train = data_file('adult', 'train_small')
    test = data_file('adult', 'test_small')
    cd = data_file('adult', 'train.cd')

    learn_pool = Pool(data=train, column_description=cd)
    test_pool = Pool(data=test, column_description=cd)

    model = CatBoostClassifier(iterations=5, loss_function='Logloss')
    model.fit(learn_pool, eval_set=test_pool)

    predictions = model.predict(test_pool)


if __name__ == '__main__':
    test_adult()
