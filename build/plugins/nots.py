import fnmatch
import os
import ytest

from _common import to_yesno, strip_roots, rootrel_arc_src


def _create_pm(unit):
    from lib.nots.package_manager import manager

    sources_path = unit.path()
    module_path = unit.get("MODDIR")
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
    lf_paths = []

    for lf_path in args:
        abs_lf_path = unit.resolve(unit.resolve_arc_path(lf_path))
        if not abs_lf_path:
            raise Exception("lockfile not found: {}".format(lf_path))
        lf_paths.append(abs_lf_path)

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
        # Use `TEST_FOR_PATH`-relative `rootDir`, since tsc will be run with `TEST_FOR_PATH`'s as curdir.
        rel_test_for_path = os.path.relpath(unit.get("TS_TEST_FOR_PATH"), strip_roots(unit.path()))
        root_dir = os.path.join(rel_test_for_path, root_dir)

    unit.set(["TS_CONFIG_ROOT_DIR", root_dir])
    unit.set(["TS_CONFIG_OUT_DIR", out_dir])
    unit.set(["TS_CONFIG_SOURCE_MAP", to_yesno(tsconfig.compiler_option("sourceMap"))])
    unit.set(["TS_CONFIG_DECLARATION", to_yesno(tsconfig.compiler_option("declaration"))])
    unit.set(["TS_CONFIG_DECLARATION_MAP", to_yesno(tsconfig.compiler_option("declarationMap"))])
    unit.set(["TS_CONFIG_PRESERVE_JSX", to_yesno(tsconfig.compiler_option("jsx") == "preserve")])

    _setup_eslint(unit)


def on_ts_test_configure(unit):
    from lib.nots.package_manager import constants

    test_runner_handlers = _get_test_runner_handlers()
    test_runner = unit.get("TS_TEST_RUNNER")

    if test_runner not in test_runner_handlers:
        raise Exception("Test runner: {} is not available, try to use one of these: {}"
                        .format(test_runner, ", ".join(test_runner_handlers.keys())))

    if not test_runner:
        raise Exception("Test runner is not specified")

    test_files = ytest.get_values_list(unit, "_TS_TEST_SRCS_VALUE")
    if not test_files:
        raise Exception("No tests found in {}".format(unit.path()))

    config_path = unit.get(unit.get("TS_TEST_CONFIG_PATH_VAR"))
    abs_config_path = unit.resolve(unit.resolve_arc_path(config_path))
    if not abs_config_path:
        raise Exception("{} config not found: {}".format(test_runner, config_path))

    pm = _create_pm(unit)
    mod_dir = unit.get("MODDIR")
    test_files = _resolve_module_files(unit, mod_dir, test_files)
    data_dirs = list(set([os.path.dirname(rootrel_arc_src(p, unit))
                          for p in (ytest.get_values_list(unit, "_TS_TEST_DATA_VALUE") or [])]))

    test_record_args = {
        "CUSTOM-DEPENDENCIES": " ".join(pm.get_peers_from_package_json()),
        "TS-TEST-FOR-PATH": unit.get("TS_TEST_FOR_PATH"),
        "TS-ROOT-DIR": unit.get("TS_CONFIG_ROOT_DIR"),
        "TS-OUT-DIR": unit.get("TS_CONFIG_OUT_DIR"),
        "TS-TEST-DATA-DIRS": ytest.serialize_list(data_dirs),
        "TS-TEST-DATA-DIRS-RENAME": unit.get("_TS_TEST_DATA_DIRS_RENAME_VALUE"),
        "NODE-MODULES-BUNDLE-FILENAME": constants.NODE_MODULES_WORKSPACE_BUNDLE_FILENAME,
        "CONFIG-PATH": config_path,
    }

    add_ts_test_type = test_runner_handlers[test_runner]
    add_ts_test_type(unit, test_runner, test_files, test_record_args)


def _get_test_runner_handlers():
    return {
        "jest": _add_jest_ts_test,
        "hermione": _add_hermione_ts_test,
    }


def _add_jest_ts_test(unit, test_runner, resolved_files, test_record_args):
    nots_plugins_path = os.path.join(unit.get("NOTS_PLUGINS_PATH"), "jest")
    test_record_args.update({
        "CUSTOM-DEPENDENCIES": " ".join((test_record_args["CUSTOM-DEPENDENCIES"], nots_plugins_path)),
        "NOTS-PLUGINS-PATH": nots_plugins_path,
    })

    _add_test_type(unit, test_runner, resolved_files, test_record_args)


def _add_hermione_ts_test(unit, test_runner, resolved_files, test_record_args):
    test_tags = list(set(["ya:fat", "ya:external"] + ytest.get_values_list(unit, "TEST_TAGS_VALUE")))
    test_requirements = list(set(["network:full"] + ytest.get_values_list(unit, "TEST_REQUIREMENTS_VALUE")))

    if not len(test_record_args["TS-TEST-DATA-DIRS"]):
        _add_default_hermione_test_data(unit, test_record_args)

    test_record_args.update({
        "SIZE": "LARGE",
        "TAG": ytest.serialize_list(test_tags),
        "REQUIREMENTS": ytest.serialize_list(test_requirements),
    })

    _add_test_type(unit, test_runner, resolved_files, test_record_args)


def _add_default_hermione_test_data(unit, test_record_args):
    mod_dir = unit.get("MODDIR")
    root_dir = test_record_args["TS-ROOT-DIR"]
    out_dir = test_record_args["TS-OUT-DIR"]
    test_for_path = test_record_args["TS-TEST-FOR-PATH"]

    abs_root_dir = os.path.normpath(os.path.join(unit.resolve(unit.path()), root_dir))
    file_paths = _find_file_paths(abs_root_dir, "**/screens/*/*/*.png")
    file_dirs = [os.path.dirname(f) for f in file_paths]

    rename_from, rename_to = [os.path.relpath(os.path.normpath(os.path.join(mod_dir, d)), test_for_path)
                              for d in [root_dir, out_dir]]

    test_record_args.update({
        "TS-TEST-DATA-DIRS": ytest.serialize_list(_resolve_module_files(unit, mod_dir, file_dirs)),
        "TS-TEST-DATA-DIRS-RENAME": "{}:{}".format(rename_from, rename_to),
    })


def _setup_eslint(unit):
    if unit.get("_NO_LINT_VALUE") == "none":
        return

    lint_files = ytest.get_values_list(unit, "_TS_LINT_SRCS_VALUE")
    if not lint_files:
        return

    from lib.nots.package_manager import constants
    pm = _create_pm(unit)
    mod_dir = unit.get("MODDIR")
    resolved_files = _resolve_module_files(unit, mod_dir, lint_files)
    test_record_args = {
        "CUSTOM-DEPENDENCIES": " ".join(pm.get_peers_from_package_json()),
        "ESLINT_CONFIG_NAME": unit.get("ESLINT_CONFIG_NAME"),
        "NODE-MODULES-BUNDLE-FILENAME": constants.NODE_MODULES_WORKSPACE_BUNDLE_FILENAME,
    }

    _add_test_type(unit, "eslint", resolved_files, test_record_args, mod_dir)


def _resolve_module_files(unit, mod_dir, file_paths):
    resolved_files = []

    for path in file_paths:
        resolved = rootrel_arc_src(path, unit)
        if resolved.startswith(mod_dir):
            resolved = resolved[len(mod_dir) + 1:]
        resolved_files.append(resolved)

    return resolved_files


def _find_file_paths(abs_path, pattern):
    file_paths = []
    _, ext = os.path.splitext(pattern)

    for root, _, filenames in os.walk(abs_path):
        if not any(f.endswith(ext) for f in filenames):
            continue

        abs_file_paths = [os.path.join(root, f) for f in filenames]

        for file_path in fnmatch.filter(abs_file_paths, pattern):
            file_paths.append(file_path)

    return file_paths


def _add_test_type(unit, test_type, test_files, test_record_args=None, test_cwd=None):
    if test_record_args is None:
        test_record_args = {}

    test_dir = ytest.get_norm_unit_path(unit)

    test_record = {
        "TEST-NAME": test_type.lower(),
        "TEST-TIMEOUT": unit.get("TEST_TIMEOUT") or "",
        "TEST-ENV": ytest.prepare_env(unit.get("TEST_ENV_VALUE")),
        "TESTED-PROJECT-NAME": os.path.splitext(unit.filename())[0],
        "SCRIPT-REL-PATH": test_type,
        "SOURCE-FOLDER-PATH": test_dir,
        "BUILD-FOLDER-PATH": test_dir,
        "BINARY-PATH": os.path.join(test_dir, unit.filename()),
        "SPLIT-FACTOR": unit.get("TEST_SPLIT_FACTOR") or "",
        "FORK-MODE": unit.get("TEST_FORK_MODE") or "",
        "SIZE": "SMALL",
        "TEST-FILES": ytest.serialize_list(test_files),
        "TEST-CWD": test_cwd or "",
        "TAG": ytest.serialize_list(ytest.get_values_list(unit, "TEST_TAGS_VALUE")),
        "REQUIREMENTS": ytest.serialize_list(ytest.get_values_list(unit, "TEST_REQUIREMENTS_VALUE")),
    }
    test_record.update(test_record_args)

    data = ytest.dump_test(unit, test_record)
    if data:
        unit.set_property(["DART_DATA", data])
        ytest.save_in_file(unit.get("TEST_DART_OUT_FILE"), data)
