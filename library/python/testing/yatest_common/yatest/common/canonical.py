import os
import logging
import shutil
import tempfile

import six

from . import process
from . import runtime
from . import path

yatest_logger = logging.getLogger("ya.test")


def _copy(src, dst, universal_lines=False):
    if universal_lines:
        with open(dst, "wb") as f:
            mode = "rbU" if six.PY2 else "rb"
            for line in open(src, mode):
                f.write(line)
        return
    shutil.copy(src, dst)


@runtime.default_arg0
def canonical_file(
    path, diff_tool=None, local=False, universal_lines=False, diff_file_name=None, diff_tool_timeout=None
):
    """
    Create canonical file that can be returned from a test
    :param path: path to the file
    :param diff_tool: custom diff tool to use for comparison with the canonical one, if None - default will be used
    :param local: save file locally, otherwise move to sandbox
    :param universal_lines: normalize EOL
    :param diff_tool_timeout: timeout for running diff tool
    :return: object that can be canonized
    """
    abs_path = os.path.abspath(path)
    assert os.path.exists(abs_path), "Canonical path {} does not exist".format(path)
    tempdir = tempfile.mkdtemp(prefix="canon_tmp", dir=runtime.build_path())
    safe_path = os.path.join(tempdir, os.path.basename(abs_path))
    # if the created file is in output_path, we copy it, so that it will be available when the tests finishes
    _copy(path, safe_path, universal_lines=universal_lines)
    if diff_tool:
        if not isinstance(diff_tool, six.string_types):
            try:  # check if iterable
                if not isinstance(diff_tool[0], six.string_types):
                    raise Exception("Invalid custom diff-tool: not cmd")
            except Exception:
                raise Exception("Invalid custom diff-tool: not binary path")
    return runtime._get_ya_plugin_instance().file(
        safe_path, diff_tool=diff_tool, local=local, diff_file_name=diff_file_name, diff_tool_timeout=diff_tool_timeout
    )


@runtime.default_arg0
def canonical_dir(path, diff_tool=None, local=False, diff_file_name=None, diff_tool_timeout=None):
    abs_path = os.path.abspath(path)
    assert os.path.exists(abs_path), "Canonical path {} does not exist".format(path)
    assert os.path.isdir(abs_path), "Path {} is not a directory".format(path)
    if diff_file_name and not diff_tool:
        raise Exception("diff_file_name can be only be used with diff_tool for canonical_dir")
    tempdir = tempfile.mkdtemp()
    safe_path = os.path.join(tempdir, os.path.basename(abs_path))
    shutil.copytree(abs_path, safe_path)
    return runtime._get_ya_plugin_instance().file(
        safe_path, diff_tool=diff_tool, local=local, diff_file_name=diff_file_name, diff_tool_timeout=diff_tool_timeout
    )


def canonical_execute(
    binary,
    args=None,
    check_exit_code=True,
    shell=False,
    timeout=None,
    cwd=None,
    env=None,
    stdin=None,
    stderr=None,
    creationflags=0,
    file_name=None,
    save_locally=False,
    close_fds=False,
    diff_tool=None,
    diff_file_name=None,
    diff_tool_timeout=None,
    data_transformer=None,
):
    """
    Shortcut to execute a binary and canonize its stdout
    :param binary: absolute path to the binary
    :param args: binary arguments
    :param check_exit_code: will raise ExecutionError if the command exits with non zero code
    :param shell: use shell to run the command
    :param timeout: execution timeout
    :param cwd: working directory
    :param env: command environment
    :param stdin: command stdin
    :param stderr: command stderr
    :param creationflags: command creation flags
    :param file_name: output file name. if not specified program name will be used
    :param diff_tool: path to custome diff tool
    :param diff_file_name: custom diff file name to create when diff is found
    :param diff_tool_timeout: timeout for running diff tool
    :param data_transformer: data modifier (before canonize)
    :return: object that can be canonized
    """
    if isinstance(binary, list):
        command = binary
    else:
        command = [binary]
    command += _prepare_args(args)
    if shell:
        command = " ".join(command)
    execute_args = locals()
    del execute_args["binary"]
    del execute_args["args"]
    del execute_args["file_name"]
    del execute_args["save_locally"]
    del execute_args["diff_tool"]
    del execute_args["diff_file_name"]
    del execute_args["diff_tool_timeout"]
    del execute_args["data_transformer"]
    if not file_name and stdin:
        file_name = os.path.basename(stdin.name)
    return _canonical_execute(
        process.execute,
        execute_args,
        file_name,
        save_locally,
        diff_tool,
        diff_file_name,
        diff_tool_timeout,
        data_transformer,
    )


def canonical_py_execute(
    script_path,
    args=None,
    check_exit_code=True,
    shell=False,
    timeout=None,
    cwd=None,
    env=None,
    stdin=None,
    stderr=None,
    creationflags=0,
    file_name=None,
    save_locally=False,
    close_fds=False,
    diff_tool=None,
    diff_file_name=None,
    diff_tool_timeout=None,
    data_transformer=None,
):
    """
    Shortcut to execute a python script and canonize its stdout
    :param script_path: path to the script arcadia relative
    :param args: script arguments
    :param check_exit_code: will raise ExecutionError if the command exits with non zero code
    :param shell: use shell to run the command
    :param timeout: execution timeout
    :param cwd: working directory
    :param env: command environment
    :param stdin: command stdin
    :param stderr: command stderr
    :param creationflags: command creation flags
    :param file_name: output file name. if not specified program name will be used
    :param diff_tool: path to custome diff tool
    :param diff_file_name: custom diff file name to create when diff is found
    :param diff_tool_timeout: timeout for running diff tool
    :param data_transformer: data modifier (before canonize)
    :return: object that can be canonized
    """
    command = [runtime.source_path(script_path)] + _prepare_args(args)
    if shell:
        command = " ".join(command)
    execute_args = locals()
    del execute_args["script_path"]
    del execute_args["args"]
    del execute_args["file_name"]
    del execute_args["save_locally"]
    del execute_args["diff_tool"]
    del execute_args["diff_file_name"]
    del execute_args["diff_tool_timeout"]
    del execute_args["data_transformer"]
    return _canonical_execute(
        process.py_execute,
        execute_args,
        file_name,
        save_locally,
        diff_tool,
        diff_file_name,
        diff_tool_timeout,
        data_transformer,
    )


def _prepare_args(args):
    if args is None:
        args = []
    if isinstance(args, six.string_types):
        args = list(map(lambda a: a.strip(), args.split()))
    return args


def _canonical_execute(
    excutor, kwargs, file_name, save_locally, diff_tool, diff_file_name, diff_tool_timeout, data_transformer
):
    res = excutor(**kwargs)
    command = kwargs["command"]
    file_name = file_name or process.get_command_name(command)
    if file_name.endswith(".exe"):
        file_name = os.path.splitext(file_name)[0]  # don't want to bring windows stuff in file names
    out_file_path = path.get_unique_file_path(runtime.output_path(), "{}.out.txt".format(file_name))
    err_file_path = path.get_unique_file_path(runtime.output_path(), "{}.err.txt".format(file_name))
    if not data_transformer:

        def data_transformer(x):
            return x

    try:
        os.makedirs(os.path.dirname(out_file_path))
    except OSError:
        pass

    with open(out_file_path, "wb") as out_file:
        yatest_logger.debug("Will store file in %s", out_file_path)
        out_file.write(data_transformer(res.std_out))

    if res.std_err:
        with open(err_file_path, "wb") as err_file:
            err_file.write(res.std_err)

    return canonical_file(
        out_file_path,
        local=save_locally,
        diff_tool=diff_tool,
        diff_file_name=diff_file_name,
        diff_tool_timeout=diff_tool_timeout,
    )
