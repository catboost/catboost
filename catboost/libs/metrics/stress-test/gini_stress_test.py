from catboost.utils import eval_metric
import numpy as np

# https://www.kaggle.com/batzner/gini-coefficient-an-intuitive-explanation

def gini(actual, pred):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)

def gini_normalized(actual, pred):
    return gini(actual, pred) / gini(actual, actual)
    
list_sizes = [10**exp for exp in range(1, 7)]

for size in list_sizes:
    print("Evaluating for dataset size =", size)
    for k in range(100):
        actual = np.random.randint(0, 2, size)
        probs = np.random.random(size)
        python_calc = gini_normalized(actual, probs)
        catboost_calc = eval_metric(actual, probs, "NormalizedGini")[0]
        if abs(python_calc - catboost_calc) > 1e-5:
            print("Error:", python_calc, catboost_calc)
            raise
    print("success")
