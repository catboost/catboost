import os
import shutil
import subprocess
import tarfile

from ..package_manager import constants
from .ts_config import TsConfig
from .ts_errors import TsCompilationError


class TscWrapper(object):
    _TSCONFIG_FILENAME = "tsconfig.json"

    def __init__(self, build_root, build_path, sources_path, nodejs_bin_path, script_path, config_path):
        self.build_root = build_root
        self.build_path = build_path
        self.sources_path = sources_path
        self.nodejs_bin_path = nodejs_bin_path
        self.script_path = script_path
        self.config_path = config_path

    def compile(self):
        self._prepare_dependencies()
        config = self._build_config()
        self._exec_tsc(["--build", config.path])

    def _prepare_dependencies(self):
        self._copy_package_json()
        self._unpack_node_modules()

    def _copy_package_json(self):
        # TODO: Validate "main" and "files" - they should include files from the output directory.
        shutil.copyfile(
            os.path.join(self.sources_path, constants.PACKAGE_JSON_FILENAME),
            os.path.join(self.build_path, constants.PACKAGE_JSON_FILENAME),
        )

    def _unpack_node_modules(self):
        nm_bundle_path = os.path.join(self.build_path, constants.NODE_MODULES_BUNDLE_FILENAME)
        if os.path.isfile(nm_bundle_path):
            with tarfile.open(nm_bundle_path) as tf:
                tf.extractall(os.path.join(self.build_path, "node_modules"))

    def _build_config(self):
        config = TsConfig.load(self.config_path)
        config.validate()
        config.transform_paths(
            build_path=self.build_path,
            sources_path=self.sources_path,
        )

        config.path = os.path.join(self.build_path, self._TSCONFIG_FILENAME)
        config.write()

        return config

    def _exec_tsc(self, args):
        p = subprocess.Popen(
            [self.nodejs_bin_path, self.script_path] + args,
            cwd=self.build_path,
            stdin=None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = p.communicate()

        if p.returncode != 0:
            raise TsCompilationError(p.returncode, stdout.decode("utf-8"), stderr.decode("utf-8"))
