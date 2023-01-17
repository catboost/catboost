import os

import ytest
from _common import to_yesno, strip_roots, rootrel_arc_src


def _create_pm(unit):
    from lib.nots.package_manager import manager

    sources_path = unit.path()
    module_path = None
    if unit.get("TS_TEST_FOR"):
        sources_path = unit.get("TS_TEST_FOR_DIR")
        module_path = unit.get("TS_TEST_FOR_PATH")

    return manager(
        sources_path=unit.resolve(sources_path),
        build_root="$B",
        build_path=unit.path().replace("$S", "$B", 1),
        contribs_path=unit.get("NPM_CONTRIBS_PATH"),
        nodejs_bin_path=None,
        script_path=None,
        module_path=module_path,
    )


def on_from_npm_lockfiles(unit, *args):
    pm = _create_pm(unit)
    lf_paths = map(lambda p: unit.resolve(unit.resolve_arc_path(p)), args)

    for pkg in pm.extract_packages_meta_from_lockfiles(lf_paths):
        unit.onfrom_npm([pkg.name, pkg.version, pkg.sky_id, pkg.integrity, pkg.integrity_algorithm, pkg.tarball_path])


def onnode_modules(unit):
    pm = _create_pm(unit)
    unit.onpeerdir(pm.get_local_peers_from_package_json())
    ins, outs = pm.calc_node_modules_inouts()
    unit.on_node_modules(["IN"] + sorted(ins) + ["OUT"] + sorted(outs))


def on_ts_configure(unit, tsconfig_path):
    from lib.nots.typescript import TsConfig

    abs_tsconfig_path = unit.resolve(unit.resolve_arc_path(tsconfig_path))
    if not abs_tsconfig_path:
        raise Exception("tsconfig not found: {}".format(tsconfig_path))

    tsconfig = TsConfig.load(abs_tsconfig_path)
    tsconfig.validate()
    root_dir = tsconfig.compiler_option("rootDir")
    out_dir = tsconfig.compiler_option("outDir")
    if unit.get("TS_TEST_FOR") == "yes":
        # Use `TEST_FOR_PATH`-relative `rootDir`, since tsc will be runned with `TEST_FOR_PATH`'s as curdir.
        rel_test_for_path = os.path.relpath(unit.get("TS_TEST_FOR_PATH"), strip_roots(unit.path()))
        root_dir = os.path.join(rel_test_for_path, root_dir)

    unit.set(["TS_CONFIG_ROOT_DIR", root_dir])
    unit.set(["TS_CONFIG_OUT_DIR", out_dir])
    unit.set(["TS_CONFIG_SOURCE_MAP", to_yesno(tsconfig.compiler_option("sourceMap"))])
    unit.set(["TS_CONFIG_DECLARATION", to_yesno(tsconfig.compiler_option("declaration"))])
    unit.set(["TS_CONFIG_DECLARATION_MAP", to_yesno(tsconfig.compiler_option("declarationMap"))])
    unit.set(["TS_CONFIG_PRESERVE_JSX", to_yesno(tsconfig.compiler_option("jsx") == "preserve")])

    _setup_eslint(unit)


def on_ts_test_configure(unit, jestconfig_path):
    from lib.nots.package_manager import constants

    test_files = ytest.get_values_list(unit, "_TS_TEST_SRCS_VALUE")
    if not test_files:
        raise Exception("No tests found in {}".format(unit.path()))

    abs_jestconfig_path = unit.resolve(unit.resolve_arc_path(jestconfig_path))
    if not abs_jestconfig_path:
        raise Exception("jest config not found: {}".format(jestconfig_path))

    pm = _create_pm(unit)
    mod_dir = unit.get("MODDIR")
    resolved_files = _resolve_test_files(unit, mod_dir, test_files)
    test_record_args = {
        "CUSTOM-DEPENDENCIES": " ".join(pm.get_peers_from_package_json()),
        "TS-TEST-FOR-PATH": unit.get("TS_TEST_FOR_PATH"),
        "TS-OUT-DIR": unit.get("TS_CONFIG_OUT_DIR"),
        "NODE-MODULES-BUNDLE-FILENAME": constants.NODE_MODULES_WORKSPACE_BUNDLE_FILENAME,
        "JEST-CONFIG-PATH": jestconfig_path,
    }

    _add_test_type(unit, "ts_test", mod_dir, resolved_files, test_record_args)


def _setup_eslint(unit):
    if unit.get('_NO_LINT_VALUE') == "none":
        return

    lint_files = ytest.get_values_list(unit, '_TS_LINT_SRCS_VALUE')
    if not lint_files:
        return

    # MODDIR == devtools/dummy_arcadia/ts/packages/with_lint
    # CURDIR == $S/MODDIR
    mod_dir = unit.get('MODDIR')
    resolved_files = _resolve_test_files(unit, mod_dir, lint_files)

    _add_eslint(unit, mod_dir, resolved_files)


def _add_eslint(unit, test_cwd, test_files):
    test_record_args = {
        'ESLINT_CONFIG_NAME': unit.get('ESLINT_CONFIG_NAME'),
    }

    _add_test_type(unit, "eslint", test_cwd, test_files, test_record_args)


def on_hermione_configure(unit, config_path):
    test_files = ytest.get_values_list(unit, '_HERMIONE_SRCS_VALUE')
    if not test_files:
        return

    mod_dir = unit.get('MODDIR')
    resolved_files = _resolve_test_files(unit, mod_dir, test_files)

    _add_hermione(unit, config_path, mod_dir, resolved_files)


def _add_hermione(unit, config_path, test_cwd, test_files):
    test_tags = list(set(['ya:fat', 'ya:external'] + ytest.get_values_list(unit, 'TEST_TAGS_VALUE')))
    test_requirements = list(set(['network:full'] + ytest.get_values_list(unit, 'TEST_REQUIREMENTS_VALUE')))

    test_record_args = {
        'SIZE': 'LARGE',
        'TAG': ytest.serialize_list(test_tags),
        'REQUIREMENTS': ytest.serialize_list(test_requirements),
        'HERMIONE-CONFIG-PATH': config_path,
    }

    _add_test_type(unit, "hermione", test_cwd, test_files, test_record_args)


def _resolve_test_files(unit, mod_dir, file_paths):
    resolved_files = []

    for path in file_paths:
        resolved = rootrel_arc_src(path, unit)
        if resolved.startswith(mod_dir):
            resolved = resolved[len(mod_dir) + 1:]
        resolved_files.append(resolved)

    return resolved_files


def _add_test_type(unit, test_type, test_cwd, test_files, test_record_args=None):
    if test_record_args is None:
        test_record_args = {}

    test_dir = ytest.get_norm_unit_path(unit)

    test_record = {
        'TEST-NAME': test_type.lower(),
        'TEST-TIMEOUT': unit.get('TEST_TIMEOUT') or '',
        'TEST-ENV': ytest.prepare_env(unit.get('TEST_ENV_VALUE')),
        'TESTED-PROJECT-NAME': os.path.splitext(unit.filename())[0],
        'SCRIPT-REL-PATH': test_type,
        'SOURCE-FOLDER-PATH': test_dir,
        'BUILD-FOLDER-PATH': test_dir,
        'BINARY-PATH': os.path.join(test_dir, unit.filename()),
        'SPLIT-FACTOR': unit.get('TEST_SPLIT_FACTOR') or '',
        'FORK-MODE': unit.get('TEST_FORK_MODE') or '',
        'SIZE': 'SMALL',
        'TEST-FILES': ytest.serialize_list(test_files),
        'TEST-CWD': test_cwd,
        'TAG': ytest.serialize_list(ytest.get_values_list(unit, 'TEST_TAGS_VALUE')),
        'REQUIREMENTS': ytest.serialize_list(ytest.get_values_list(unit, 'TEST_REQUIREMENTS_VALUE')),
    }
    test_record.update(test_record_args)

    data = ytest.dump_test(unit, test_record)
    if data:
        unit.set_property(['DART_DATA', data])
        ytest.save_in_file(unit.get('TEST_DART_OUT_FILE'), data)
