import numpy as np
from catboost.utils import eval_metric

def fairloss(actual, pred):
    x = abs(np.array(actual) - np.array(pred))
    c = 2
    res = np.mean(c**2 * ((x / c) - np.log((x/c) + 1)))
    return res

list_sizes = [10**exp for exp in range(1, 7)]

for size in list_sizes:
    print("Evaluating for dataset size =", size)
    for k in range(100):
        actual = np.random.randint(0, 2, size)
        probs = np.random.random(size)
        python_calc = fairloss(actual, probs)
        catboost_calc = eval_metric(actual, probs, "FairLoss")[0]
        if abs(python_calc - catboost_calc) > 1e-5:
            print("Error:", python_calc, catboost_calc)
            raise
    print("success")