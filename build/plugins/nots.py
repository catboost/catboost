import os.path

import ytest
from _common import to_yesno, rootrel_arc_src


def _create_pm(unit):
    from lib.nots.package_manager import manager

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

    from lib.nots.typescript import TsConfig

    tsconfig = TsConfig.load(abs_tsconfig_path)
    tsconfig.validate()

    unit.set(["TS_CONFIG_ROOT_DIR", tsconfig.compiler_option("rootDir")])
    unit.set(["TS_CONFIG_OUT_DIR", tsconfig.compiler_option("outDir")])
    unit.set(["TS_CONFIG_SOURCE_MAP", to_yesno(tsconfig.compiler_option("sourceMap"))])
    unit.set(["TS_CONFIG_DECLARATION", to_yesno(tsconfig.compiler_option("declaration"))])
    unit.set(["TS_CONFIG_DECLARATION_MAP", to_yesno(tsconfig.compiler_option("declarationMap"))])
    unit.set(["TS_CONFIG_PRESERVE_JSX", to_yesno(tsconfig.compiler_option("jsx") == "preserve")])

    _setup_eslint(unit)


def _setup_eslint(unit):
    if unit.get('LINT_LEVEL_VALUE') == "none":
        return

    lint_files = ytest.get_values_list(unit, '_TS_LINT_SRCS_VALUE')
    if not lint_files:
        return

    # MODDIR == devtools/dummy_arcadia/ts/packages/with_lint
    # CURDIR == $S/MODDIR
    mod_dir = unit.get('MODDIR')

    resolved_files = []
    for path in lint_files:
        resolved = rootrel_arc_src(path, unit)
        if resolved.startswith(mod_dir):
            resolved = resolved[len(mod_dir) + 1:]
        resolved_files.append(resolved)

    # devtools/dummy_arcadia/ts/packages/with_lint == MODDIR
    test_dir = ytest.get_norm_unit_path(unit)

    test_record = {
        'TEST-NAME': "eslint",
        'TEST-TIMEOUT': '',
        'SCRIPT-REL-PATH': "eslint",
        'TESTED-PROJECT-NAME': os.path.basename(test_dir),
        'SOURCE-FOLDER-PATH': test_dir,
        'SPLIT-FACTOR': unit.get('TEST_SPLIT_FACTOR') or '',
        'FORK-MODE': unit.get('TEST_FORK_MODE') or '',
        'SIZE': 'SMALL',
        'TEST-FILES': ytest.serialize_list(resolved_files),
        'ESLINT_CONFIG_NAME': unit.get('ESLINT_CONFIG_NAME'),
    }

    data = ytest.dump_test(unit, test_record)
    if data:
        unit.set_property(['DART_DATA', data])
        ytest.save_in_file(unit.get('TEST_DART_OUT_FILE'), data)
