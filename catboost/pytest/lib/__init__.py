import os
import pytest
import re
import tempfile
import time
import yatest.common
import yatest.common.network
import yatest.common.runtime
from .common_helpers import *  # noqa
import zipfile

from testpath.tempdir import TemporaryDirectory


def get_catboost_binary_path():
    return yatest.common.binary_path("catboost/app/catboost")


def data_file(*path):
    return yatest.common.source_path(os.path.join("catboost", "pytest", "data", *path))


@yatest.common.misc.lazy
def get_cuda_setup_error():
    for flag in yatest.common.runtime._get_ya_config().option.flags:
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


# params is either dict or iterable
# devices used only if task_type == 'GPU'
def execute_catboost(mode, task_type, params, devices='0', stdout=None, timeout=None, env=None):
    if task_type not in ('CPU', 'GPU'):
        raise Exception('task_type must be "CPU" or "GPU"')

    cmd = [
        get_catboost_binary_path(),
        mode,
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

    mkl_cbwr_env = dict(env) if env else dict()
    mkl_cbwr_env.update(MKL_CBWR='SSE4_2')
    yatest.common.execute(cmd, stdout=stdout, timeout=timeout, env=mkl_cbwr_env)


def execute_catboost_fit(task_type, params, devices='0', stdout=None, timeout=None, env=None):
    execute_catboost('fit', task_type, params, devices, stdout, timeout, env)


# cd_path could be None (and should be for yt-search-proto pools)
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

    yatest.common.execute(calc_cmd)


def local_canonical_file(*args, **kwargs):
    return yatest.common.canonical_file(*args, local=True, **kwargs)


def execute_catboost_dist(mode, cmd):
    hosts_path = yatest.common.test_output_path('hosts.txt')
    with yatest.common.network.PortManager() as pm:
        port0 = pm.get_port()
        port1 = pm.get_port()
        with open(hosts_path, 'w') as hosts:
            hosts.write('localhost:' + str(port0) + '\n')
            hosts.write('localhost:' + str(port1) + '\n')

        catboost_path = yatest.common.binary_path("catboost/app/catboost")
        worker0 = yatest.common.execute((catboost_path, 'run-worker', '--node-port', str(port0),), wait=False)
        worker1 = yatest.common.execute((catboost_path, 'run-worker', '--node-port', str(port1),), wait=False)
        while pm.is_port_free(port0) or pm.is_port_free(port1):
            time.sleep(1)

        execute_catboost(
            mode,
            'CPU',
            cmd + ('--node-type', 'Master', '--file-with-hosts', hosts_path,)
        )
        worker0.wait()
        worker1.wait()


def execute_dist_train(cmd):
    execute_catboost_dist('fit', cmd)


@pytest.fixture(scope="module")
def compressed_data():
    data_path = yatest.common.source_path(os.path.join("catboost", "pytest", "data"))
    tmp_dir = TemporaryDirectory()
    for file_name in os.listdir(data_path):
        if file_name.endswith('.zip'):
            with zipfile.ZipFile(os.path.join(data_path, file_name)) as zip_file:
                zip_file.extractall(path=tmp_dir.name)

    return tmp_dir
