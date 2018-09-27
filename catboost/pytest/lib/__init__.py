import json
import os
import pytest
import random
import re
import shutil
import tempfile
import yatest.common
import yatest.yt


def get_catboost_binary_path():
    return yatest.common.binary_path("catboost/app/catboost")


def append_params_to_cmdline(cmd, params):
    if isinstance(params, dict):
        for param in params.items():
            key = "{}".format(param[0])
            value = "{}".format(param[1])
            cmd.append(key)
            cmd.append(value)
    else:
        for param in params:
            cmd.append(param)


def data_file(*path):
    return yatest.common.source_path(os.path.join("catboost", "pytest", "data", *path))


@yatest.common.misc.lazy
def get_cuda_setup_error():
    for flag in pytest.config.option.flags:
        if re.match('HAVE_CUDA=(0|no|false)', flag, flags=re.IGNORECASE):
            return flag

    train = tempfile.NamedTemporaryFile(delete=False)
    train.write('\n'.join(['%i\t%i' % (x, x + 1) for x in range(10)]) + '\n')
    train.close()
    cd = tempfile.NamedTemporaryFile(delete=False)
    cd.write('0\tTarget\n')
    cd.close()
    try:
        cmd = (get_catboost_binary_path(), 'fit',
               '--task-type', 'GPU',
               '--devices', '0',
               '-i', '2',
               '-f', train.name,
               '--column-description', cd.name
               )
        yatest.common.execute(cmd)
    except Exception as e:
        for reason in ['GPU support was not compiled', 'CUDA driver version is insufficient']:
            if reason in str(e):
                return reason
        return str(e)
    finally:
        os.unlink(train.name)
        os.unlink(cd.name)

    return None


def execute(*args, **kwargs):
    input_data = kwargs.pop('input_data', None)
    output_data = kwargs.pop('output_data', None)

    task_gpu = 'GPU' in args[0]
    cuda_setup_error = get_cuda_setup_error() if task_gpu else None

    if task_gpu and cuda_setup_error:
        cuda_explicitly_disabled = 'HAVE_CUDA' in cuda_setup_error
        if cuda_explicitly_disabled:
            pytest.xfail(reason=cuda_setup_error)
        return yatest.yt.execute(
            *args,
            task_spec={'gpu_limit': 1},
            operation_spec={'pool_trees': ['gpu']},
            input_data=input_data,
            output_data=output_data,
            # required for quantized-marked input filenames
            data_mine_strategy=yatest.yt.process.replace_mine_strategy,
            **kwargs
        )
    return yatest.common.execute(*args, **kwargs)


# params is either dict or iterable
# devices used only if task_type == 'GPU'
def execute_catboost_fit(task_type, params, devices='0', input_data=None, output_data=None):
    if task_type not in ('CPU', 'GPU'):
        raise Exception('task_type must be "CPU" or "GPU"')

    cmd = [
        get_catboost_binary_path(),
        'fit',
        '--task-type', task_type
    ]

    if isinstance(params, dict):
        for param in params.items():
            key = "{}".format(param[0])
            value = "{}".format(param[1])
            cmd.append(key)
            cmd.append(value)
    else:
        cmd.extend(params)

    if task_type == 'GPU':
        cmd.extend(
            [
                '--devices', devices,
                '--gpu-ram-part', '0.25'
            ]
        )

    execute(cmd, input_data=input_data, output_data=output_data)


# cd_path should be None for yt-search-proto pools
def apply_catboost(model_file, pool_path, cd_path, eval_file, output_columns=None, has_header=False):
    calc_cmd = (
        get_catboost_binary_path(),
        'calc',
        '--input-path', pool_path,
        '-m', model_file,
        '--output-path', eval_file,
        '--prediction-type', 'RawFormulaVal'
    )
    if cd_path:
        calc_cmd += ('--column-description', cd_path)
    if output_columns:
        calc_cmd += ('--output-columns', ','.join(output_columns))
    if has_header:
        calc_cmd += ('--has-header',)
    execute(calc_cmd)


def get_limited_precision_dsv_diff_tool(diff_limit, have_header=False):
    diff_tool = [
        yatest.common.binary_path("catboost/tools/limited_precision_dsv_diff/limited_precision_dsv_diff"),
    ]
    if diff_limit is not None:
        diff_tool += ['--diff-limit', str(diff_limit)]
    if have_header:
        diff_tool += ['--have-header']
    return diff_tool


def local_canonical_file(*args, **kwargs):
    return yatest.common.canonical_file(*args, local=True, **kwargs)


def remove_time_from_json(filename):
    with open(filename) as f:
        log = json.load(f)
    iterations = log['iterations']
    for i, iter_info in enumerate(iterations):
        del iter_info['remaining_time']
        del iter_info['passed_time']
    with open(filename, 'w') as f:
        json.dump(log, f)
    return filename


# rewinds dst_stream to the start of the captured output so you can read it
class DelayedTee(object):

    def __init__(self, src_stream, dst_stream):
        self.src_stream = src_stream
        self.dst_stream = dst_stream

    def __enter__(self):
        self.src_stream.flush()
        self._old_src_stream = os.dup(self.src_stream.fileno())
        self._old_dst_stream_pos = self.dst_stream.tell()
        os.dup2(self.dst_stream.fileno(), self.src_stream.fileno())

    def __exit__(self, exc_type, exc_value, traceback):
        self.src_stream.flush()
        os.dup2(self._old_src_stream, self.src_stream.fileno())
        self.dst_stream.seek(self._old_dst_stream_pos)
        shutil.copyfileobj(self.dst_stream, self.src_stream)
        self.dst_stream.seek(self._old_dst_stream_pos)


binary_path = yatest.common.binary_path
test_output_path = yatest.common.test_output_path


def permute_dataset_columns(test_pool_path, cd_path, seed=123):
    permuted_test_path = yatest.common.test_output_path('permuted_test')
    permuted_cd_path = yatest.common.test_output_path('permuted_cd')
    generator = random.Random(seed)
    column_count = len(open(test_pool_path).readline().split('\t'))
    permutation = list(range(column_count))
    generator.shuffle(permutation)
    with open(cd_path) as original_cd, open(permuted_cd_path, 'w') as permuted_cd:
        for line in original_cd:
            line = line.strip()
            if not line:
                continue
            index, rest = line.split('\t', 1)
            permuted_cd.write('{}\t{}\n'.format(permutation.index(int(index)), rest))
    with open(test_pool_path) as test_pool, open(permuted_test_path, 'w') as permuted_test:
        for line in test_pool:
            splitted = line.strip().split('\t')
            permuted_test.write('\t'.join([splitted[i] for i in permutation]) + '\n')

    return permuted_test_path, permuted_cd_path
