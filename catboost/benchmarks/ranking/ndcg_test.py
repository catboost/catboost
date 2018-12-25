import numpy as np
from ndcg_kaggle import ndcg_score
from utils import ndcg


def generate_relevances(size=10):
    rel_true = np.random.randint(0, 5, size=size)
    rel_pred = np.random.uniform(0, 5, size=size)
    return rel_true, rel_pred

def test():
    for i in range(10000):
        size = np.random.randint(3, 30)
        top = np.random.randint(2, size)
        rel_true, rel_pred = generate_relevances(size)

        expect = ndcg_score([rel_true], [rel_pred], top)
        real = ndcg(rel_pred, rel_true, top)

        if np.abs(real - expect) > 1e-6:
            print('real ' + str(real) + ' but expect ' + str(expect))
            print('top ' + str(top))
            print(rel_true)
            print(rel_pred)
            raise Exception()
    print('Success')


if __name__ == "__main__":
    test()
