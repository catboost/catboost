import os
import shutil

from abc import ABCMeta, abstractmethod
from six import add_metaclass, iteritems

from ..package_manager import extract_node_modules, utils as pm_utils


@add_metaclass(ABCMeta)
class BaseBuilder(object):
    def __init__(self, build_root, build_path, sources_path, nodejs_bin_path, node_modules_bundle_path=None, copy_package_json=True, output_node_modules_path=None, external_dependencies=None):
        """
        :param build_root: build root ($ARCADIA_BUILD_ROOT)
        :type build_root: str
        :param build_path: module build path ($BINDIR)
        :type build_path: str
        :param sources_path: module sources path ($CURDIR)
        :type sources_path: str
        :param nodejs_bin_path: path to nodejs bin
        :type nodejs_bin_path: str
        :param node_modules_bundle_path: path to node_modules.tar bundle
        :type node_modules_bundle_path: str
        :param copy_package_json: whether package.json should be copied to build path
        :type copy_package_json: bool
        :param output_node_modules_path: path to re-export (copy) node_modules.tar bundle
        :type output_node_modules_path: str
        :param external_dependencies: external dependencies which will be linked to node_modules/ (mapping name → path)
        :type external_dependencies: dict
        """
        self.build_root = build_root
        self.build_path = build_path
        self.sources_path = sources_path
        self.nodejs_bin_path = nodejs_bin_path
        self.node_modules_bundle_path = node_modules_bundle_path
        self.copy_package_json = copy_package_json
        self.output_node_modules_path = output_node_modules_path
        self.external_dependencies = external_dependencies

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
        self._link_external_dependencies()

        if not os.path.isfile(self.node_modules_bundle_path):
            return

        extract_node_modules(
            build_root=self.build_root,
            bundle_path=self.node_modules_bundle_path,
            node_modules_path=pm_utils.build_nm_path(os.path.dirname(self.node_modules_bundle_path)),
        )

        if self.output_node_modules_path:
            os.rename(self.node_modules_bundle_path, self.output_node_modules_path)

    def _link_external_dependencies(self):
        nm_path = pm_utils.build_nm_path(self.build_path)
        try:
            os.makedirs(nm_path)
        except OSError:
            pass

        # Don't want to use `os.makedirs(exists_ok=True)` here (we don't want to skip all "file exists" errors).
        scope_paths = set()

        for name, src in iteritems(self.external_dependencies):
            dst = os.path.join(nm_path, name)
            scope_path = os.path.dirname(dst)
            if scope_path and scope_path not in scope_paths:
                os.mkdir(scope_path)
                scope_paths.add(scope_path)

            os.symlink(src, dst, target_is_directory=True)

    @abstractmethod
    def _build(self):
        pass
