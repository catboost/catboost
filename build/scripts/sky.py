import logging
import os
import subprocess

import fetch_from


class UnsupportedProtocolException(Exception):
    pass


def executable_path():
    return "/usr/local/bin/sky"


def is_avaliable():
    if not os.path.exists(executable_path()):
        return False
    try:
        subprocess.check_output([executable_path(), "--version"])
        return True
    except subprocess.CalledProcessError:
        return False
    except OSError:
        return False


def fetch(skynet_id, file_name, timeout=None):
    if not is_avaliable():
        raise UnsupportedProtocolException("Skynet is not available")

    target_dir = os.path.abspath(fetch_from.uniq_string_generator())
    os.mkdir(target_dir)

    cmd_args = [executable_path(), "get", "-N", "Backbone", "--user", "--wait", "--dir", target_dir, skynet_id]
    if timeout is not None:
        cmd_args += ["--timeout", str(timeout)]

    logging.info("Call skynet with args: %s", cmd_args)
    stdout = subprocess.check_output(cmd_args).strip()
    logging.debug("Skynet call with args %s is finished, result is %s", cmd_args, stdout)

    return os.path.join(target_dir, file_name)
