import os
import shutil
import yaml

from six import iteritems

from ..base import PackageJson, BasePackageManager, PackageManagerError
from .lockfile import PnpmLockfile
from .workspace import PnpmWorkspace
from .utils import build_pj_path, build_lockfile_path, build_ws_config_path, build_nm_bundle_path


class PnpmPackageManager(BasePackageManager):
    _STORE_NM_PATH = os.path.join(".pnpm", "store")
    _VSTORE_NM_PATH = os.path.join(".pnpm", "virtual-store")
    _STORE_VER = "v3"

    def install(self):
        """
        Creates node_modules directory according to the lockfile.
        """
        self._prepare_workspace()
        self._exec_command([
            "install",
            "--offline",
            "--frozen-lockfile",
            "--store-dir", self._nm_path(self._STORE_NM_PATH),
            "--virtual-store-dir", self._nm_path(self._VSTORE_NM_PATH),
            "--no-verify-store-integrity",
            "--package-import-method", "hardlink",
            "--ignore-pnpmfile",
            "--ignore-scripts",
            "--strict-peer-dependencies",
        ])
        self._fix_stores_in_modules_yaml()

    def get_peer_paths_from_package_json(self):
        """
        Returns paths of direct workspace dependencies (source root related).
        :rtype: list of str
        """
        pj = PackageJson.load(build_pj_path(self.sources_path))

        return map(lambda x: os.path.normpath(os.path.join(self.module_path, x[1])), pj.get_workspace_dep_paths())

    def calc_node_modules_inouts(self):
        """
        Returns input and output paths for command that creates `node_modules` bundle.
        :return: Pair of input and output paths with correct roots ($S or $B).
        :rtype: (list of str, list of str)
        """
        # Inputs: source package.json and lockfile, built package.jsons, lockfiles and workspace configs of deps, tarballs.
        ins = []
        # Source lockfiles are used only to get tarballs info.
        src_lf_paths = [build_lockfile_path(self.sources_path)]
        pj = PackageJson.load(build_pj_path(self.sources_path))

        for [dep_src_path, (dep_pj, depth)] in iteritems(pj.get_workspace_map()):
            if dep_src_path == self.sources_path:
                continue
            dep_mod_path = dep_src_path[len(self.sources_root) + 1:]
            # pnpm requires all package.jsons.
            ins.append(build_pj_path(dep_mod_path))
            dep_lf_src_path = build_lockfile_path(dep_src_path)
            if not os.path.isfile(dep_lf_src_path):
                continue
            src_lf_paths.append(dep_lf_src_path)
            # Merged workspace configs and lockfiles of direct deps.
            if depth == 1:
                ins.append(build_ws_config_path(dep_mod_path))
                ins.append(build_lockfile_path(dep_mod_path))

        for pkg in self.extract_packages_meta_from_lockfiles(src_lf_paths):
            ins.append(self._contrib_tarball_path(pkg))

        s_root = lambda x: os.path.join("$S", x)
        b_root = lambda x: os.path.join("$B", x)

        ins = map(b_root, ins) + [
            s_root(build_pj_path(self.module_path)),
            s_root(build_lockfile_path(self.module_path)),
        ]

        # Outputs: patched lockfile, generated workspace config, created node_modules bundle.
        outs = [b_root(f(self.module_path)) for f in (build_lockfile_path, build_ws_config_path, build_nm_bundle_path)]

        return (ins, outs)

    def extract_packages_meta_from_lockfiles(self, lf_paths):
        """
        :type lf_paths: iterable of BaseLockfile
        :rtype: iterable of LockfilePackageMeta
        """
        tarballs = set()

        for lf_path in lf_paths:
            try:
                for pkg in PnpmLockfile.load(lf_path).get_packages_meta():
                    if pkg.tarball_path not in tarballs:
                        tarballs.add(pkg.tarball_path)
                        yield pkg
            except Exception as e:
                raise PackageManagerError("Unable to process lockfile {}: {}".format(lf_path, e))

    def _prepare_workspace(self):
        pj = self._build_package_json()
        ws = PnpmWorkspace(build_ws_config_path(self.build_path))
        ws.set_from_package_json(pj)
        dep_paths = ws.get_paths()
        self._build_merged_workspace_config(ws, dep_paths)
        self._build_merged_lockfile(dep_paths)

    def _build_package_json(self):
        """
        :rtype: PackageJson
        """
        in_pj_path = build_pj_path(self.sources_path)
        out_pj_path = build_pj_path(self.build_path)
        shutil.copyfile(in_pj_path, out_pj_path)

        return PackageJson.load(out_pj_path)

    def _build_merged_lockfile(self, dep_paths):
        """
        :type dep_paths: list of str
        :rtype: PnpmLockfile
        """
        in_lf_path = build_lockfile_path(self.sources_path)
        out_lf_path = build_lockfile_path(self.build_path)

        lf = PnpmLockfile.load(in_lf_path)
        # Change to the output path for correct path calcs on merging.
        lf.path = out_lf_path

        for dep_path in dep_paths:
            if dep_path is self.build_path:
                continue
            lf_path = build_lockfile_path(dep_path)
            if os.path.isfile(lf_path):
                lf.merge(PnpmLockfile.load(lf_path))

        lf.update_tarball_resolutions(lambda p: self._contrib_tarball_url(p))
        lf.write()

    def _build_merged_workspace_config(self, ws, dep_paths):
        """
        :type ws: PnpmWorkspaceConfig
        :type dep_paths: list of str
        """
        for dep_path in dep_paths:
            if dep_path is self.build_path:
                continue
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
            data["storeDir"] = os.path.join(self.sources_path, "node_modules", self._STORE_NM_PATH, self._STORE_VER)
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
