import os
import shutil
import subprocess
import tarfile

from ..package_manager import constants
from .ts_config import TsConfig
from .ts_errors import TsCompilationError


class TsBundleWrapper(object):
    def __init__(self, build_root, build_path, sources_path, nodejs_bin_path, script_path, ts_config_path, webpack_config_path, webpack_resource):
        self.build_root = build_root
        self.build_path = build_path
        self.sources_path = sources_path
        self.nodejs_bin_path = nodejs_bin_path
        self.script_path = script_path
        self.ts_config_curpath = ts_config_path
        self.ts_config_binpath = os.path.join(build_path, os.path.basename(ts_config_path))
        self.webpack_config_curpath = webpack_config_path
        self.webpack_config_binpath = os.path.join(build_path, os.path.basename(webpack_config_path))
        self.webpack_resource = webpack_resource

    def compile(self):
        self._prepare_dependencies()
        self._build_configs()
        self._exec_webpack()
        self._pack_bundle()

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

    def _build_configs(self):
        shutil.copyfile(
            self.webpack_config_curpath,
            self.webpack_config_binpath
        )

        config = TsConfig.load(self.ts_config_curpath)
        config.validate()
        config.transform_paths(
            build_path=self.build_path,
            sources_path=self.sources_path,
        )

        config.path = self.ts_config_binpath
        config.write()

    def _exec_webpack(self):
        custom_envs = {
            "WEBPACK_CONFIG": self.webpack_config_binpath,
            "CURDIR": self.sources_path,
            "BINDIR": self.build_path,
            "NODE_MODULES_DIRS": self.webpack_resource
        }

        p = subprocess.Popen(
            [self.nodejs_bin_path, self.script_path, "--config", self.webpack_config_binpath],
            cwd=self.build_path,
            stdin=None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=custom_envs,
        )
        stdout, stderr = p.communicate()

        if p.returncode != 0:
            raise TsCompilationError(p.returncode, stdout.decode("utf-8"), stderr.decode("utf-8"))

    def _pack_bundle(self):
        with tarfile.open(self.build_path + "/bundle.tar", "w") as tf:
            tf.add(self.build_path + "/bundle")
            tf.close()
