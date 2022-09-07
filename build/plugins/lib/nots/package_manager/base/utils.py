import os

from .constants import PACKAGE_JSON_FILENAME, NODE_MODULES_DIRNAME, NODE_MODULES_BUNDLE_FILENAME


def build_pj_path(p):
    return os.path.join(p, PACKAGE_JSON_FILENAME)


def build_nm_path(p):
    return os.path.join(p, NODE_MODULES_DIRNAME)


def build_nm_bundle_path(p):
    return os.path.join(p, NODE_MODULES_BUNDLE_FILENAME)
