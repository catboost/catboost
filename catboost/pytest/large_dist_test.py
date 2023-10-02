import yatest.common
import pytest

from lib import (
    data_file,
    execute_dist_train,
    local_canonical_file,
)

CATBOOST_PATH = yatest.common.binary_path("catboost/app/catboost")
SCORE_CALC_OBJ_BLOCK_SIZES = ['60', '5000000']
SCORE_CALC_OBJ_BLOCK_SIZES_IDS = ['calc_block=60', 'calc_block=5000000']


@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_dist_train_many_trees(dev_score_calc_obj_block_size):
    pool_path = data_file('higgs', 'train_small')
    test_path = data_file('higgs', 'test_small')
    cd_path = data_file('higgs', 'train.cd')
    cmd = (
        '--loss-function', 'Logloss',
        '-f', pool_path,
        '-t', test_path,
        '--column-description', cd_path,
        '-i', '1000',
        '-w', '0.03',
        '-T', '4',
        '--random-strength', '0',
        '--has-time',
        '--bootstrap-type', 'No',
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
    )

    eval_path = yatest.common.test_output_path('test.eval')
    execute_dist_train(cmd + ('--eval-file', eval_path,))

    return [local_canonical_file(eval_path)]
