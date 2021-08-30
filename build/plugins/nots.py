import os

from _common import to_yesno
from lib.nots.package_manager import manager
from lib.nots.typescript import TsConfig


def _create_pm(unit):
    return manager(
        sources_path=unit.resolve(unit.path()),
        build_root="$B",
        build_path=unit.path().replace("$S", "$B", 1),
        contribs_path=unit.get("NPM_CONTRIBS_PATH"),
        nodejs_bin_path=None,
        script_path=None,
    )


def on_from_npm_lockfiles(unit, *args):
    lf_paths = map(lambda p: unit.resolve(unit.resolve_arc_path(p)), args)

    for pkg in _create_pm(unit).extract_packages_meta_from_lockfiles(lf_paths):
        unit.onfrom_npm([pkg.name, pkg.version, pkg.sky_id, pkg.integrity, pkg.integrity_algorithm, pkg.tarball_path])


def onnode_modules(unit):
    pm = _create_pm(unit)
    unit.onpeerdir(pm.get_peer_paths_from_package_json())
    ins, outs = pm.calc_node_modules_inouts()
    unit.on_node_modules(["IN"] + sorted(ins) + ["OUT"] + sorted(outs))


def on_ts_configure(unit, tsconfig_path):
    abs_tsconfig_path = unit.resolve(unit.resolve_arc_path(tsconfig_path))
    if not abs_tsconfig_path:
        raise Exception("tsconfig not found: {}".format(tsconfig_path))

    tsconfig = TsConfig.load(abs_tsconfig_path)
    tsconfig.validate()

    unit.set(["TS_CONFIG_ROOT_DIR", tsconfig.compiler_option("rootDir")])
    unit.set(["TS_CONFIG_OUT_DIR", tsconfig.compiler_option("outDir")])
    unit.set(["TS_CONFIG_SOURCE_MAP", to_yesno(tsconfig.compiler_option("sourceMap"))])
    unit.set(["TS_CONFIG_DECLARATION", to_yesno(tsconfig.compiler_option("declaration"))])
    unit.set(["TS_CONFIG_DECLARATION_MAP", to_yesno(tsconfig.compiler_option("declarationMap"))])
    unit.set(["TS_CONFIG_PRESERVE_JSX", to_yesno(tsconfig.compiler_option("jsx") == "preserve")])
