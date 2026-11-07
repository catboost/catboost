"""Shared host APIs needed by GPU vector training; no estimator is fitted."""
import os

import numpy as np
import pytest
from catboost.utils import eval_metric


pytestmark = pytest.mark.skipif(
    os.environ.get('CATBOOST_NATIVE_METAL_TESTS') != '1',
    reason='requires the rebuilt CatBoost host metric correction',
)


@pytest.mark.parametrize('objective',['MultiLogloss','MultiCrossEntropy'])
@pytest.mark.parametrize('rows',[33,128,10003])
@pytest.mark.parametrize('thread_count',[1,4])
def test_fractional_weights_and_parallel_metric_denominators(objective,rows,thread_count):
    rng=np.random.default_rng(871+rows)
    raw=rng.normal(size=(rows,3))
    labels=rng.uniform(size=(rows,3))
    if objective=='MultiLogloss':labels=(labels>.5).astype(float)
    weights=rng.uniform(.05,1.3,rows).astype(np.float32)
    weights[::13]=0
    expected=np.average(np.mean(np.logaddexp(0,raw)-labels*raw,axis=1),weights=weights.astype(float))
    actual=eval_metric(labels,raw,objective,weight=weights,thread_count=thread_count)[0]
    assert actual==pytest.approx(expected,rel=3e-8,abs=3e-9)
