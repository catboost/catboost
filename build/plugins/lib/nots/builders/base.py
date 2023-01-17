import os
import shutil

from abc import ABCMeta, abstractmethod
from six import add_metaclass

from ..package_manager import extract_node_modules, utils as pm_utils


@add_metaclass(ABCMeta)
class BaseBuilder(object):
    def __init__(self, build_root, build_path, sources_path, nodejs_bin_path, node_modules_bundle_path, copy_package_json, output_node_modules_path):
        self.build_root = build_root
        self.build_path = build_path
        self.sources_path = sources_path
        self.nodejs_bin_path = nodejs_bin_path
        self.node_modules_bundle_path = node_modules_bundle_path
        self.copy_package_json = copy_package_json
        self.output_node_modules_path = output_node_modules_path

    def build(self):
        self._copy_package_json()
        self._prepare_dependencies()
        self._build()

    def _copy_package_json(self):
        if not self.copy_package_json:
            return

        shutil.copyfile(
            pm_utils.build_pj_path(self.sources_path),
            pm_utils.build_pj_path(self.build_path),
        )

    def _prepare_dependencies(self):
        if not os.path.isfile(self.node_modules_bundle_path):
            return

        extract_node_modules(
            build_root=self.build_root,
            bundle_path=self.node_modules_bundle_path,
            node_modules_path=pm_utils.build_nm_path(os.path.dirname(self.node_modules_bundle_path)),
        )

        if self.output_node_modules_path:
            os.rename(self.node_modules_bundle_path, self.output_node_modules_path)

    @abstractmethod
    def _build(self):
        pass
