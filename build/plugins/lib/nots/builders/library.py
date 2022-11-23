import os
import subprocess

from .base import BaseBuilder
from ..package_manager import constants
from ..typescript import TsConfig, TsCompilationError


class LibraryBuilder(BaseBuilder):
    def __init__(self, build_root, build_path, sources_path, nodejs_bin_path, node_modules_bundle_path, tsc_resource, ts_config_path, plugins=[], typings=[]):
        super(LibraryBuilder, self).__init__(
            build_root=build_root,
            build_path=build_path,
            sources_path=sources_path,
            nodejs_bin_path=nodejs_bin_path,
            node_modules_bundle_path=node_modules_bundle_path,
            copy_package_json=True,
            output_node_modules_path=constants.NODE_MODULES_WORKSPACE_BUNDLE_FILENAME,
            external_dependencies=dict([t.split(":") for t in typings]),
        )
        self.tsc_resource = tsc_resource
        self.script_path = os.path.join(tsc_resource, "typescript", "lib", "tsc.js")
        self.ts_config_curpath = ts_config_path
        self.ts_config_binpath = os.path.join(build_path, os.path.basename(ts_config_path))
        self.plugins = plugins

    def _build(self):
        self._prepare_bindir()
        self._exec_tsc()

    def _prepare_bindir(self):
        ts_config = self._create_ts_config()
        ts_root_dir = ts_config.compiler_option("rootDir")
        os.symlink(
            os.path.join(self.sources_path, ts_root_dir),
            os.path.join(self.build_path, ts_root_dir),
        )

    def _create_ts_config(self):
        ts_config = TsConfig.load(self.ts_config_curpath)
        for p in self.plugins:
            ts_config.inject_plugin({"transform": p})
        ts_config.path = self.ts_config_binpath
        ts_config.write()

        return ts_config

    def _exec_tsc(self):
        env = {
            "NODE_PATH": "{}:{}".format(self.tsc_resource, "node_modules"),
        }

        p = subprocess.Popen(
            [self.nodejs_bin_path, self.script_path, "--build", self.ts_config_binpath],
            cwd=self.build_path,
            env=env,
            stdin=None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = p.communicate()

        if p.returncode != 0:
            raise TsCompilationError(p.returncode, stdout.decode("utf-8"), stderr.decode("utf-8"))
