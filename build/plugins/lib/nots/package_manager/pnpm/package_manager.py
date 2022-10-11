import os
import yaml

from six import iteritems

from ..base import BasePackageManager, PackageManagerError
from ..base.utils import build_pj_path, build_nm_path, build_nm_bundle_path, s_rooted, b_rooted
from ..base.node_modules_bundler import bundle_node_modules
from ..base.constants import NODE_MODULES_BUNDLE_FILENAME
from .lockfile import PnpmLockfile
from .workspace import PnpmWorkspace
from .utils import build_lockfile_path, build_ws_config_path


class PnpmPackageManager(BasePackageManager):
    _STORE_NM_PATH = os.path.join(".pnpm", "store")
    _VSTORE_NM_PATH = os.path.join(".pnpm", "virtual-store")
    _STORE_VER = "v3"

    @classmethod
    def load_lockfile(cls, path):
        """
        :param path: path to lockfile
        :type path: str
        :rtype: PnpmLockfile
        """
        return PnpmLockfile.load(path)

    @classmethod
    def load_lockfile_from_dir(cls, dir_path):
        """
        :param dir_path: path to directory with lockfile
        :type dir_path: str
        :rtype: PnpmLockfile
        """
        return cls.load_lockfile(build_lockfile_path(dir_path))

    def create_node_modules(self):
        """
        Creates node_modules directory according to the lockfile.
        """
        ws = self._prepare_workspace()
        self._exec_command([
            "install",
            "--offline",
            "--frozen-lockfile",
            "--public-hoist-pattern", "",
            "--store-dir", self._nm_path(self._STORE_NM_PATH),
            "--virtual-store-dir", self._nm_path(self._VSTORE_NM_PATH),
            "--no-verify-store-integrity",
            "--package-import-method", "hardlink",
            "--ignore-pnpmfile",
            "--ignore-scripts",
            "--strict-peer-dependencies",
        ])
        self._fix_stores_in_modules_yaml()

        bundle_node_modules(
            build_root=self.build_root,
            node_modules_path=self._nm_path(),
            peers=ws.get_paths(base_path=self.module_path, ignore_self=True),
            bundle_path=NODE_MODULES_BUNDLE_FILENAME,
        )

    def calc_node_modules_inouts(self):
        """
        Returns input and output paths for command that creates `node_modules` bundle.
        Inputs:
            - source package.json and lockfile,
            - built package.jsons of all deps,
            - merged lockfiles and workspace configs of direct non-leave deps,
            - tarballs.
        Outputs:
            - merged lockfile,
            - generated workspace config,
            - created node_modules bundle.
        :rtype: (list of str, list of str)
        """
        ins = [
            s_rooted(build_pj_path(self.module_path)),
            s_rooted(build_lockfile_path(self.module_path)),
        ]
        outs = [
            b_rooted(build_lockfile_path(self.module_path)),
            b_rooted(build_ws_config_path(self.module_path)),
            b_rooted(build_nm_bundle_path(self.module_path)),
        ]

        # Source lockfiles are used only to get tarballs info.
        src_lf_paths = [build_lockfile_path(self.sources_path)]
        pj = self.load_package_json_from_dir(self.sources_path)

        for [dep_src_path, (_, depth)] in iteritems(pj.get_workspace_map(ignore_self=True)):
            dep_mod_path = dep_src_path[len(self.sources_root) + 1:]
            # pnpm requires all package.jsons.
            ins.append(b_rooted(build_pj_path(dep_mod_path)))

            dep_lf_src_path = build_lockfile_path(dep_src_path)
            if not os.path.isfile(dep_lf_src_path):
                # It is ok for leaves.
                continue
            src_lf_paths.append(dep_lf_src_path)

            if depth == 1:
                ins.append(b_rooted(build_ws_config_path(dep_mod_path)))
                ins.append(b_rooted(build_lockfile_path(dep_mod_path)))

        for pkg in self.extract_packages_meta_from_lockfiles(src_lf_paths):
            ins.append(b_rooted(self._contrib_tarball_path(pkg)))

        return (ins, outs)

    def extract_packages_meta_from_lockfiles(self, lf_paths):
        """
        :type lf_paths: iterable of BaseLockfile
        :rtype: iterable of LockfilePackageMeta
        """
        tarballs = set()

        for lf_path in lf_paths:
            try:
                for pkg in self.load_lockfile(lf_path).get_packages_meta():
                    if pkg.tarball_path not in tarballs:
                        tarballs.add(pkg.tarball_path)
                        yield pkg
            except Exception as e:
                raise PackageManagerError("Unable to process lockfile {}: {}".format(lf_path, e))

    def _prepare_workspace(self):
        """
        :rtype: PnpmWorkspace
        """
        pj = self._build_package_json()
        ws = PnpmWorkspace(build_ws_config_path(self.build_path))
        ws.set_from_package_json(pj)
        dep_paths = ws.get_paths(ignore_self=True)
        self._build_merged_workspace_config(ws, dep_paths)
        self._build_merged_lockfile(dep_paths)

        return ws

    def _build_package_json(self):
        """
        :rtype: PackageJson
        """
        pj = self.load_package_json_from_dir(self.sources_path)
        # Change to the output path for correct path for workspace.
        pj.path = build_pj_path(self.build_path)
        pj.write()

        return pj

    def _build_merged_lockfile(self, dep_paths):
        """
        :type dep_paths: list of str
        :rtype: PnpmLockfile
        """
        lf = self.load_lockfile_from_dir(self.sources_path)
        # Change to the output path for correct path calcs on merging.
        lf.path = build_lockfile_path(self.build_path)

        for dep_path in dep_paths:
            lf_path = build_lockfile_path(dep_path)
            if os.path.isfile(lf_path):
                lf.merge(self.load_lockfile(lf_path))

        lf.update_tarball_resolutions(lambda p: self._contrib_tarball_url(p))
        lf.write()

    def _build_merged_workspace_config(self, ws, dep_paths):
        """
        NOTE: This method mutates `ws`.
        :type ws: PnpmWorkspaceConfig
        :type dep_paths: list of str
        """
        for dep_path in dep_paths:
            ws_config_path = build_ws_config_path(dep_path)
            if os.path.isfile(ws_config_path):
                ws.merge(PnpmWorkspace.load(ws_config_path))

        ws.write()

    def _fix_stores_in_modules_yaml(self):
        """
        Ensures that store paths are the same as would be after installing deps in the source dir.
        This is required to reuse `node_modules` after build.
        """
        with open(self._nm_path(".modules.yaml"), "r+") as f:
            data = yaml.load(f, Loader=yaml.CSafeLoader)
            # NOTE: pnpm requires absolute store path here.
            data["storeDir"] = os.path.join(build_nm_path(self.sources_path), self._STORE_NM_PATH, self._STORE_VER)
            data["virtualStoreDir"] = self._VSTORE_NM_PATH
            f.seek(0)
            yaml.dump(data, f, Dumper=yaml.CSafeDumper)
            f.truncate()

    def _get_default_options(self):
        return super(PnpmPackageManager, self)._get_default_options() + [
            "--stream",
            "--reporter", "append-only",
            "--no-color",
        ]

    def _get_debug_log_path(self):
        return self._nm_path(".pnpm-debug.log")
