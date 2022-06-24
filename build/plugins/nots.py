import os.path

import ytest
from _common import to_yesno, resolve_common_const
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

    _setup_eslint(unit)


def _setup_eslint(unit):
    # MODDIR == devtools/dummy_arcadia/ts/packages/with_lint
    # CURDIR == $S/MODDIR
    mod_dir = unit.get('MODDIR')

    lint_files = ytest.get_values_list(unit, '_TS_LINT_SRCS_VALUE')
    resolved_files = []
    for path in lint_files:
        resolved = unit.resolve(unit.resolve_arc_path(resolve_common_const(path)))
        resolved_files.append(resolved)

    if resolved_files:
        # ESLint should start in the MODDIR to properly use relative paths in config files
        _add_eslint(unit, mod_dir, resolved_files)


def _add_eslint(unit, test_cwd, test_files):
    check_type = "eslint"
    test_dir = ytest.get_norm_unit_path(unit)

    test_record = {
        'TEST-NAME': check_type.lower(),
        'TEST-TIMEOUT': '',
        'SCRIPT-REL-PATH': check_type,
        'TESTED-PROJECT-NAME': os.path.basename(test_dir),
        'SOURCE-FOLDER-PATH': test_dir,
        'SPLIT-FACTOR': unit.get('TEST_SPLIT_FACTOR') or '',
        'FORK-MODE': unit.get('TEST_FORK_MODE') or '',
        'SIZE': 'SMALL',
        'TEST-FILES': ytest.serialize_list(test_files),
        'TEST-CWD': test_cwd,
    }

    data = ytest.dump_test(unit, test_record)
    if data:
        unit.set_property(['DART_DATA', data])
        ytest.save_in_file(unit.get('TEST_DART_OUT_FILE'), data)
