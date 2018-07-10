import os
import re
import json
import shlex
import logging
import tempfile
import subprocess
import collections

import yatest.common as ytc

logger = logging.getLogger(__name__)


class InvalidInputError(Exception):
    pass


# Don't forget to sync changes in the interface and defaults with yatest.common.execute
def execute(
    command, check_exit_code=True,
    shell=False, timeout=None,
    cwd=None, env=None,
    stdin=None, stdout=None, stderr=None,
    creationflags=0, wait=True,
    process_progress_listener=None, close_fds=False,
    collect_cores=True, check_sanitizer=True, preexec_fn=None, on_timeout=None,
    # YT specific
    input_data=None, output_data=None,
    data_mine_strategy=None,
    operation_spec=None, task_spec=None,
    yt_proxy=None, output_result_path=None,
):
    """
    Executes a command on the YT. Listed below are options whose behavior is different from yatest.common.execute
    :param command: can be a list of arguments or a string
    :param timeout: timeout for command executed on the YT (doesn't take into account the time spent for execution preparation - uploading/downloading data, etc)
    :param cwd: ignored
    :param env: all paths matched prefixes yatest.common.*_path will be fixed
    :param stdin: stdin will be fully read before execution and uploaded to the YT
    :param stdout: stdout will be available after the execution of the command on the YT. Set to False to skip downloading
    :param stderr: same as stdout
    :param process_progress_listener: ignored
    :param preexec_fn: ignored
    :param on_timeout: ignored
    :param input_data: map of input files/dirs required for command run which will be uploaded to YT (local path -> YT sandbox path)
    :param output_data: map of output files/dirs which will be downloaded from YT after command execution (YT sandbox path -> local path)
                        Take into account that runner will call os.path.dirname(YT sandbox path) to create intermediate directories for every entry
    :param data_mine_strategy: allows to provide own function to mine input data and fix cmd. For more info take a look at *_mine_strategy()
    :param op_spec: YT operation spec
    :param task_spec: YT task spec
    :param output_result_path: specify path to output archive. Used for test purposes
    :return: Execution object
    """
    test_tool_bin = _get_test_tool_bin()

    if not wait:
        raise NotImplementedError()

    env = _get_fixed_env(env)

    orig_command = command
    command, to_upload, to_download = _fix_user_data(command, shell, input_data, output_data, data_mine_strategy or default_mine_strategy)
    command_name = ytc.process.get_command_name(command)

    exec_spec = {
        'env': env,
        'command': command,
        'timeout': timeout,
        'input_data': to_upload,
        'output_data': to_download,
    }

    if stdin:
        if isinstance(stdin, basestring):
            stdin_path = stdin
        else:
            logger.deubg('Reading stdin')
            with tempfile.NamedTemporaryFile(delete=False) as afile:
                afile.write(stdin.read())
                stdin_path = afile.name
        to_upload[stdin_path] = 'env/stdin'
        exec_spec['stdin'] = 'env/stdin'

    for stream, name in [
        (True, 'meta'),
        (stdout, 'stdout'),
        (stderr, 'stderr'),
    ]:
        if stream is not False:
            path = 'env/{}'.format(name)
            exec_spec[name] = path
            to_download[path] = ytc.get_unique_file_path(ytc.work_path(), 'yt_vanilla_{}_{}'.format(command_name, name))

    runner_log_dst = 'env/runner_log'
    exec_spec['runner_log'] = runner_log_dst
    to_download[runner_log_dst] = ytc.path.get_unique_file_path(ytc.output_path(), 'yt_vanilla_wrapper_{}.log'.format(command_name))

    exec_spec['op_spec'] = _get_spec(
        default={'max_failed_job_count': 2},
        user=operation_spec,
    )
    exec_spec['task_spec'] = _get_spec(
        default={'memory_limit': 3 * (1024 ** 3)},
        user=task_spec,
        mandatory={'job_count': 1},
    )

    exec_spec_path = _dump_spec(exec_spec)

    executor_cmd = [
        test_tool_bin, 'yt_vanilla_execute',
        '--spec-file', exec_spec_path,
        '--log-path', ytc.path.get_unique_file_path(ytc.output_path(), 'yt_vanilla_op_{}.log'.format(command_name)),
    ]
    if yt_proxy:
        executor_cmd += ['--yt-proxy', yt_proxy]
    if output_result_path:
        executor_cmd += ['--output-path', output_result_path]

    res = ytc.execute(executor_cmd, collect_cores=collect_cores, wait=False, check_sanitizer=check_sanitizer)
    if wait:
        res.wait(check_exit_code=True)

    _patch_result(res, exec_spec, orig_command, stdout, stderr, check_exit_code, timeout)
    return res


def default_mine_strategy(arg, prefix, replacement):
    if arg.startswith(prefix):
        path = replacement + arg[len(prefix):]
        # (fixed argument for command, path to local file, fixed path to file for YT wrapper, is it input data)
        return path, arg, path, os.path.exists(arg)


def replace_mine_strategy(arg, prefix, replacement):
    if prefix in arg:
        match = re.match("(.*?)({})(.*)".format(re.escape(prefix)), arg)
        fixed_arg = match.group(1) + replacement + match.group(3)
        local_path = match.group(2) + match.group(3)
        remote_path = replacement + match.group(3)
        return fixed_arg, local_path, remote_path, os.path.exists(local_path)


def _patch_result(result, exec_spec, command, user_stdout, user_stderr, check_exit_code, timeout):

    def local_path(name):
        return exec_spec['output_data'][exec_spec[name]]

    if user_stdout is not False:
        with open(local_path('stdout')) as afile:
            result._std_out = afile.read()

    if user_stderr is not False:
        with open(local_path('stderr')) as afile:
            result._std_err = afile.read()

    result._command = command

    with open(local_path('meta')) as afile:
        meta = json.load(afile)

    result._elapsed = meta['elapsed']
    result._metrics = meta['metrics']
    result._exit_code = meta['exit_code']

    if meta['timeout']:
        raise ytc.ExecutionTimeoutError(result, "{} second(s) wait timeout has expired".format(timeout))

    # if rc != 0 and check_exit_code - _finalise will print stderr and stdout
    if not check_exit_code or result._exit_code == 0:
        logger.debug("Command over YT output:\n%s", ytc.process.truncate(result._std_out, ytc.process.MAX_OUT_LEN))
        logger.debug("Command over YT errors:\n%s", ytc.process.truncate(result._std_err, ytc.process.MAX_OUT_LEN))

    result._finalise(check_exit_code)


def _get_test_tool_bin():
    msg = 'You can use yatest.yt.execute only from tests'

    try:
        import pytest
    except ImportError:
        raise Exception(msg)

    if not pytest.config.test_tool_bin:
        raise Exception(msg)

    return pytest.config.test_tool_bin


def _get_spec(default=None, user=None, mandatory=None):
    spec = default or {}
    spec.update(user or {})
    spec.update(mandatory or {})
    return default


@ytc.misc.lazy
def _get_output_replace_map():
    d = collections.OrderedDict()
    # ytc.test_output_path() is based on output_path()
    d[ytc.output_path()] = 'env/output_path'
    return d


@ytc.misc.lazy
def _get_input_replace_map():
    tmpdir = ytc.misc.first(os.environ.get(var) for var in ['TMPDIR', 'TEMP', 'TMP']) or '/tmp'
    # order matters - source_path is build_path's subdir
    d = collections.OrderedDict()
    d[ytc.source_path()] = 'env/source_path'
    d[ytc.data_path()] = 'env/data_path'
    # yatest.common.binary_path() based on build_path()
    d[ytc.build_path()] = 'env/build_path'
    d[tmpdir] = 'env/tempdir'
    return d


@ytc.misc.lazy
def _get_replace_map():
    # order matters - output dirs is input's subdir
    d = _get_output_replace_map().copy().copy()
    d.update(_get_input_replace_map())
    return d


def _get_fixed_env(env):
    if env is None:
        env = os.environ.copy()

    for env_var in [
        'HOME',
        'PORT_SYNC_PATH',
        'PWD',
        'PYTHONPATH',
        'USER',
        'YT_TOKEN',
    ]:
        if env_var in env:
            del env[env_var]

    def fix_path(p):
        for prefix, val in _get_replace_map().iteritems():
            if p.startswith(prefix):
                return val + p[len(prefix):]
        return p

    return {k: fix_path(v) for k, v in env.iteritems()}


def _fix_user_data(orig_cmd, shell, user_input, user_output, strategy):
    cmd = []
    input_data, output_data = {}, {}

    if isinstance(orig_cmd, basestring):
        orig_cmd = shlex.split(orig_cmd)

    def check_arg(arg):
        for prefix, val in _get_replace_map().iteritems():
            res = strategy(arg, prefix, val)
            if res:
                fixed_arg, local_path, remote_path, inlet = res
                if inlet:
                    input_data.update({local_path: remote_path})
                else:
                    output_data.update({remote_path: local_path})
                cmd.append(fixed_arg)
                return True

        return False

    for arg in orig_cmd:
        if check_arg(arg):
            continue
        cmd.append(arg)

    for srcs, dst in [
        (user_input, input_data),
        (user_output, output_data),
    ]:
        if srcs:
            for path in srcs.values():
                if path.startswith('/'):
                    raise InvalidInputError("Don't use abs path for specifying destination path '{}'".format(path))
            dst.update(srcs)

    return subprocess.list2cmdline(cmd) if shell else cmd, input_data, output_data


def _dump_spec(data):
    filename = tempfile.NamedTemporaryFile(delete=False)
    with open(filename.name, 'w') as afile:
        json.dump(data, afile, indent=4, sort_keys=True)
    return filename.name
