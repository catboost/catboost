import subprocess
import sys
import os

import _common as common


class CustomCommand(object):
    def __setstate__(self, sdict):
        if isinstance(sdict, tuple):
            for elem in sdict:
                if isinstance(elem, dict):
                    for key in elem:
                        setattr(self, key, elem[key])

        self._source_root = None
        self._build_root = None

    def set_source_root(self, path):
        self._source_root = path

    def set_build_root(self, path):
        self._build_root = path

    def call(self, args, **kwargs):
        cwd = self._get_call_specs('cwd', kwargs)
        stdout_path = self._get_call_specs('stdout', kwargs)

        resolved_args = []

        for arg in args:
            resolved_args.append(self.resolve_path(arg))

        if stdout_path:
            stdout = open(stdout_path, 'wb')
        else:
            stdout = None

        env = os.environ.copy()
        env['ASAN_OPTIONS'] = 'detect_leaks=0'

        rc = subprocess.call(resolved_args, cwd=cwd, stdout=stdout, env=env)

        if stdout:
            stdout.close()
        if rc:
            sys.exit(rc)

    def resolve_path(self, path):
        return common.resolve_to_abs_path(path, self._source_root, self._build_root)

    def _get_call_specs(self, name, kwargs):
        if isinstance(kwargs, dict):
            param = kwargs.get(name, None)
            if param:
                return self.resolve_path(param)
        return None


def addrule(*unused):
    pass


def addparser(*unused, **kwargs):
    pass
