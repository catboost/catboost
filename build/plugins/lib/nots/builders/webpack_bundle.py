import os
import shutil
import subprocess
import tarfile

from .base import BaseBuilder
from ..typescript import TsConfig
from ..package_manager import constants


class WebpackBundlingError(RuntimeError):
    def __init__(self, code, stdout, stderr):
        self.code = code
        self.stdout = stdout
        self.stderr = stderr

        super(WebpackBundlingError, self).__init__("webpack exited with code {}:\n{}\n{}".format(code, stdout, stderr))


class WebpackBundleBuilder(BaseBuilder):
    def __init__(self, build_root, build_path, sources_path, nodejs_bin_path, node_modules_bundle_path, webpack_resource, webpack_config_path, ts_config_path):
        super(WebpackBundleBuilder, self).__init__(
            build_root=build_root,
            build_path=build_path,
            sources_path=sources_path,
            nodejs_bin_path=nodejs_bin_path,
            node_modules_bundle_path=node_modules_bundle_path,
            copy_package_json=True,
            output_node_modules_path=constants.NODE_MODULES_WORKSPACE_BUNDLE_FILENAME,
        )
        self.webpack_resource = webpack_resource
        self.script_path = os.path.join(webpack_resource, ".bin", "webpack")
        self.webpack_config_curpath = webpack_config_path
        self.webpack_config_binpath = os.path.join(build_path, os.path.basename(webpack_config_path))
        self.ts_config_curpath = ts_config_path
        self.ts_config_binpath = os.path.join(build_path, os.path.basename(ts_config_path))

    def _build(self):
        self._prepare_bindir()
        self._exec_webpack()
        self._bundle()

    def _prepare_bindir(self):
        shutil.copyfile(
            self.webpack_config_curpath,
            self.webpack_config_binpath
        )

        ts_config = TsConfig.load(self.ts_config_curpath)
        ts_config.write(self.ts_config_binpath)
        ts_root_dir = ts_config.compiler_option("rootDir")
        os.symlink(
            os.path.join(self.sources_path, ts_root_dir),
            os.path.join(self.build_path, ts_root_dir),
        )

    def _exec_webpack(self):
        env = {
            "NODE_PATH": "{}:{}".format(self.webpack_resource, "node_modules"),
        }

        p = subprocess.Popen(
            [self.nodejs_bin_path, self.script_path, "--config", self.webpack_config_binpath],
            cwd=self.build_path,
            stdin=None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        stdout, stderr = p.communicate()

        if p.returncode != 0:
            raise WebpackBundlingError(p.returncode, stdout.decode("utf-8"), stderr.decode("utf-8"))

    def _bundle(self):
        with tarfile.open(os.path.join(self.build_path, "bundle.tar"), "w") as tf:
            tf.add(os.path.join(self.build_path, "bundle"))
            tf.close()
