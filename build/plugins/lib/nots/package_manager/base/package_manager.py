import os
import sys
import subprocess

from abc import ABCMeta, abstractmethod
from six import add_metaclass

from .constants import NPM_REGISTRY_URL
from .package_json import PackageJson
from .utils import build_nm_path, build_pj_path


class PackageManagerError(RuntimeError):
    pass


class PackageManagerCommandError(PackageManagerError):
    def __init__(self, cmd, code, stdout, stderr):
        self.cmd = cmd
        self.code = code
        self.stdout = stdout
        self.stderr = stderr

        msg = "package manager exited with code {} while running {}:\n{}\n{}".format(code, cmd, stdout, stderr)
        super(PackageManagerCommandError, self).__init__(msg)


@add_metaclass(ABCMeta)
class BasePackageManager(object):
    def __init__(self, build_root, build_path, sources_path, nodejs_bin_path, script_path, contribs_path, module_path=None, sources_root=None):
        self.module_path = build_path[len(build_root) + 1:] if module_path is None else module_path
        self.build_path = build_path
        self.sources_path = sources_path
        self.build_root = build_root
        self.sources_root = sources_path[:-len(self.module_path) - 1] if sources_root is None else sources_root
        self.nodejs_bin_path = nodejs_bin_path
        self.script_path = script_path
        self.contribs_path = contribs_path

    @classmethod
    def load_package_json(cls, path):
        """
        :param path: path to package.json
        :type path: str
        :rtype: PackageJson
        """
        return PackageJson.load(path)

    @classmethod
    def load_package_json_from_dir(cls, dir_path):
        """
        :param dir_path: path to directory with package.json
        :type dir_path: str
        :rtype: PackageJson
        """
        return cls.load_package_json(build_pj_path(dir_path))

    @classmethod
    @abstractmethod
    def load_lockfile(cls, path):
        pass

    @classmethod
    @abstractmethod
    def load_lockfile_from_dir(cls, dir_path):
        pass

    @abstractmethod
    def create_node_modules(self):
        pass

    @abstractmethod
    def calc_node_modules_inouts(self):
        pass

    @abstractmethod
    def extract_packages_meta_from_lockfiles(self, lf_paths):
        pass

    def get_local_peers_from_package_json(self):
        """
        Returns paths of direct workspace dependencies (source root related).
        :rtype: list of str
        """
        return self.load_package_json_from_dir(self.sources_path).get_workspace_dep_paths(base_path=self.module_path)

    def get_peers_from_package_json(self):
        """
        Returns paths of workspace dependencies (source root related).
        :rtype: list of str
        """
        pj = self.load_package_json_from_dir(self.sources_path)
        prefix_len = len(self.sources_root) + 1

        return [p[prefix_len:] for p in pj.get_workspace_map(ignore_self=True).keys()]

    def _exec_command(self, args, include_defaults=True):
        if not self.nodejs_bin_path:
            raise PackageManagerError("Unable to execute command: nodejs_bin_path is not configured")

        cmd = [self.nodejs_bin_path, self.script_path] + args + (self._get_default_options() if include_defaults else [])
        p = subprocess.Popen(
            cmd,
            cwd=self.build_path,
            stdin=None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = p.communicate()

        if p.returncode != 0:
            self._dump_debug_log()

            raise PackageManagerCommandError(cmd, p.returncode, stdout.decode("utf-8"), stderr.decode("utf-8"))

    def _nm_path(self, *parts):
        return os.path.join(build_nm_path(self.build_path), *parts)

    def _contrib_tarball_path(self, pkg):
        return os.path.join(self.contribs_path, pkg.tarball_path)

    def _contrib_tarball_url(self, pkg):
        return "file:" + self._contrib_tarball_path(pkg)

    def _get_default_options(self):
        return ["--registry", NPM_REGISTRY_URL]

    def _get_debug_log_path(self):
        return None

    def _dump_debug_log(self):
        log_path = self._get_debug_log_path()

        if not log_path:
            return

        try:
            with open(log_path) as f:
                sys.stderr.write("Package manager log {}:\n{}\n".format(log_path, f.read()))
        except Exception:
            sys.stderr.write("Failed to dump package manager log {}.\n".format(log_path))
