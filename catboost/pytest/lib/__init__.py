import os
import pytest
import re
import tempfile
import yatest.common
import yatest.yt
from common_helpers import *  # noqa


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


def run_nvidia_smi():
    import subprocess
    subprocess.call(['/usr/bin/nvidia-smi'])


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
            task_spec={
                # temporary layers
                'layer_paths': [
                    '//home/codecoverage/nvidia-396.tar.gz',
                    '//porto_layers/ubuntu-xenial-base.tar.xz',
                ],
                'gpu_limit': 1
            },
            operation_spec={
                'pool_trees': ['gpu_geforce_1080ti'],
                'scheduling_tag_filter': 'porto',
            },
            input_data=input_data,
            output_data=output_data,
            # required for quantized-marked input filenames
            data_mine_strategy=yatest.yt.process.replace_mine_strategy,
            # required for debug purposes
            init_func=run_nvidia_smi,
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
def apply_catboost(model_file, pool_path, cd_path, eval_file, output_columns=None, has_header=False, args=None):
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
    if args:
        calc_cmd += tuple(args.strip().split())

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


def format_crossvalidation(is_inverted, n, k):
    cv_type = 'Inverted' if is_inverted else 'Classical'
    return '{}:{};{}'.format(cv_type, n, k)
