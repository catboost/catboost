import os
import pytest
import re
import yatest.common as yc
from catboost_pytest_lib import (
    data_file,
    get_limited_precision_dsv_diff_tool,
)

CATBOOST_PATH = yc.binary_path("catboost/app/catboost")


def diff_tool(threshold=None):
    return get_limited_precision_dsv_diff_tool(threshold, True)


def get_test_output_path(pattern):
    assert '{}' in pattern
    result = yc.test_output_path(pattern.format(''))
    attempt = 0
    while os.path.exists(result):
        result = yc.test_output_path(pattern.format(attempt))
        attempt += 1
    return result


class Dataset(object):
    def __init__(self, data_dir):
        self.train_file = data_file(data_dir, 'train_small')
        self.test_file = data_file(data_dir, 'test_small')
        self.cd_file = data_file(data_dir, 'train.cd')


class TestModeNormalizeModel(object):
    def fit(self, loss_function, dataset):
        model_file = get_test_output_path('model{}.bin')
        yc.execute([
            CATBOOST_PATH,
            'fit',
            '--loss-function', loss_function,
            '-f', dataset.train_file,
            '--cd', dataset.cd_file,
            '-i', '10',
            '-T', '4',
            '-m', model_file,
        ])
        return model_file

    def normalize_model(self, model, dataset, *pools):
        normalized_model = get_test_output_path('normalized_model{}.bin')
        yc.execute([
            CATBOOST_PATH,
            'normalize-model',
            '-m', model,
            '--output-model', normalized_model,
            '--cd', dataset.cd_file,
            '-T', '4',
        ] + sum([['-i', getattr(dataset, pool)] for pool in pools], []))
        return normalized_model

    def eval_model(self, model, dataset, pool):
        eval_result = get_test_output_path('eval_result{}.txt')
        yc.execute([
            CATBOOST_PATH,
            'calc',
            '-m', model,
            '--input-path', getattr(dataset, pool),
            '--cd', dataset.cd_file,
            '--output-path', eval_result,
            '-T', '4',
            '--output-columns', 'RawFormulaVal',
        ])
        return eval_result

    def get_scale_bias(self, model):
        scale_bias_txt = get_test_output_path('scale_bias{}.txt')
        with open(scale_bias_txt, 'wt') as to_scale_bias_txt:
            yc.execute([
                CATBOOST_PATH,
                'normalize-model',
                '-m', model,
                '--print-scale-and-bias',
            ], stdout=to_scale_bias_txt)
        for line in open(scale_bias_txt).readlines():
            m = re.match(r'Input model scale (\S+) bias (\S+)', line)
            if m:
                return float(m.group(1)), float(m.group(2))
        raise ValueError('No scale/bias in model {}'.format(model))

    def set_scale_bias(self, model, scale, bias, output_model_format):
        model_with_set_scale_bias = get_test_output_path('model_with_set_scale_bias{}.' + output_model_format)
        yc.execute([
            CATBOOST_PATH,
            'normalize-model',
            '-m', model,
            '--output-model', model_with_set_scale_bias,
            '--output-model-format', output_model_format,
            '--set-scale', scale,
            '--set-bias', bias,
        ])
        return model_with_set_scale_bias

    def get_minmax(self, eval_result):
        data = map(float, open(eval_result).readlines()[1:])
        return min(data), max(data)

    def test_normalize_good(self):
        dataset = Dataset('adult')
        model = self.fit('RMSE', dataset)
        normalized_model = self.normalize_model(model, dataset, 'test_file')
        normalized_eval = self.eval_model(normalized_model, dataset, 'test_file')
        normalized_minmax = self.get_minmax(normalized_eval)
        assert normalized_minmax == (0, 1)

    def test_normalize_bad(self):
        dataset = Dataset('precipitation_small')
        model = self.fit('MultiClass', dataset)
        with pytest.raises(yc.process.ExecutionError, match='normaliz.*multiclass'):
            self.normalize_model(model, dataset, 'test_file')

    def test_normalize_idempotent(self):
        dataset = Dataset('adult')
        model = self.fit('Logloss', dataset)
        model_normalized_once = self.normalize_model(model, dataset, 'test_file', 'train_file')
        model_normalized_twice = self.normalize_model(model_normalized_once, dataset, 'test_file', 'train_file')
        eval1 = self.eval_model(model_normalized_once, dataset, 'test_file')
        eval2 = self.eval_model(model_normalized_twice, dataset, 'test_file')
        yc.execute(get_limited_precision_dsv_diff_tool(0) + [eval1, eval2])

    def test_sum(self):
        dataset = Dataset('adult')
        model = self.fit('Logloss', dataset)
        model1 = self.normalize_model(model, dataset, 'test_file')
        model2 = self.normalize_model(model, dataset, 'train_file')
        s1, b1 = self.get_scale_bias(model1)
        s2, b2 = self.get_scale_bias(model2)

        # Pick weights w1, w2 so that model_sum = w1 * model1 + w2 * model2
        det = (s1 * b2 - s2 * b1)
        w1 = b2 / det if det else 0.5
        w2 = -b1 / det if det else 0.5

        model_sum = get_test_output_path('model_sum{}.bin')
        yc.execute([
            CATBOOST_PATH,
            'model-sum',
            '--model-with-weight', '{}={}'.format(model1, w1),
            '--model-with-weight', '{}={}'.format(model2, w2),
            '--output-path', model_sum,
        ])
        eval_orig = self.eval_model(model, dataset, 'test_file')
        eval_sum = self.eval_model(model_sum, dataset, 'test_file')
        yc.execute(get_limited_precision_dsv_diff_tool(1e-8) + [eval_orig, eval_sum])

    def test_export_scale_bias(self):
        dataset = Dataset('higgs')
        model = self.fit('RMSE', dataset)
        scale, bias = '0.1234566', '0.6654322'
        for format in ['Cpp', 'Json', 'Python']:
            text_model = self.set_scale_bias(model, scale, bias, format)
            assert scale in open(text_model).read(), 'Missing scale in {}'.format(text_model)
            assert bias in open(text_model).read(), 'Missing bias in {}'.format(text_model)
