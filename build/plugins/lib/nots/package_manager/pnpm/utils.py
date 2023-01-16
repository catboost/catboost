import os

from ..base.constants import PACKAGE_JSON_FILENAME, PNPM_LOCKFILE_FILENAME, PNPM_WS_FILENAME, NODE_MODULES_BUNDLE_FILENAME


def build_pj_path(p):
    return os.path.join(p, PACKAGE_JSON_FILENAME)


def build_lockfile_path(p):
    return os.path.join(p, PNPM_LOCKFILE_FILENAME)


def build_ws_config_path(p):
    return os.path.join(p, PNPM_WS_FILENAME)


def build_nm_bundle_path(p):
    return os.path.join(p, NODE_MODULES_BUNDLE_FILENAME)
