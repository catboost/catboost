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

    unit.onsrcs(tsconfig.get_extended_paths())

    unit.set(["TS_CONFIG_ROOT_DIR", root_dir])
    unit.set(["TS_CONFIG_OUT_DIR", out_dir])
    unit.set(["TS_CONFIG_SOURCE_MAP", to_yesno(tsconfig.compiler_option("sourceMap"))])
    unit.set(["TS_CONFIG_DECLARATION", to_yesno(tsconfig.compiler_option("declaration"))])
    unit.set(["TS_CONFIG_DECLARATION_MAP", to_yesno(tsconfig.compiler_option("declarationMap"))])
    unit.set(["TS_CONFIG_PRESERVE_JSX", to_yesno(tsconfig.compiler_option("jsx") == "preserve")])

    _set_nodejs_root(unit)
    _setup_eslint(unit)


def _is_tests_enabled(unit):
    if unit.get("TIDY") == "yes":
        return False

    return True


def on_ts_test_configure(unit):
    if not _is_tests_enabled(unit):
        return

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

    mod_dir = unit.get("MODDIR")
    test_files = _resolve_module_files(unit, mod_dir, test_files)
    data_dirs = list(set([os.path.dirname(rootrel_arc_src(p, unit))
                          for p in (ytest.get_values_list(unit, "_TS_TEST_DATA_VALUE") or [])]))

    deps = _create_pm(unit).get_peers_from_package_json()
    test_record = {
        "TS-TEST-FOR-PATH": unit.get("TS_TEST_FOR_PATH"),
        "TS-ROOT-DIR": unit.get("TS_CONFIG_ROOT_DIR"),
        "TS-OUT-DIR": unit.get("TS_CONFIG_OUT_DIR"),
        "TS-TEST-DATA-DIRS": ytest.serialize_list(data_dirs),
        "TS-TEST-DATA-DIRS-RENAME": unit.get("_TS_TEST_DATA_DIRS_RENAME_VALUE"),
        "CONFIG-PATH": config_path,
    }

    add_ts_test = test_runner_handlers[test_runner]
    add_ts_test(unit, test_runner, test_files, deps, test_record)


def _get_test_runner_handlers():
    return {
        "jest": _add_jest_ts_test,
        "hermione": _add_hermione_ts_test,
    }


def _add_jest_ts_test(unit, test_runner, test_files, deps, test_record):
    nots_plugins_path = os.path.join(unit.get("NOTS_PLUGINS_PATH"), "jest")
    deps.append(nots_plugins_path)
    test_record["NOTS-PLUGINS-PATH"] = nots_plugins_path

    _add_test(unit, test_runner, test_files, deps, test_record)


def _add_hermione_ts_test(unit, test_runner, test_files, deps, test_record):
    test_tags = list(set(["ya:fat", "ya:external"] + ytest.get_values_list(unit, "TEST_TAGS_VALUE")))
    test_requirements = list(set(["network:full"] + ytest.get_values_list(unit, "TEST_REQUIREMENTS_VALUE")))

    if not len(test_record["TS-TEST-DATA-DIRS"]):
        _add_default_hermione_test_data(unit, test_record)

    test_record.update({
        "SIZE": "LARGE",
        "TAG": ytest.serialize_list(test_tags),
        "REQUIREMENTS": ytest.serialize_list(test_requirements),
    })

    _add_test(unit, test_runner, test_files, deps, test_record)


def _add_default_hermione_test_data(unit, test_record):
    mod_dir = unit.get("MODDIR")
    root_dir = test_record["TS-ROOT-DIR"]
    out_dir = test_record["TS-OUT-DIR"]
    test_for_path = test_record["TS-TEST-FOR-PATH"]

    abs_root_dir = os.path.normpath(os.path.join(unit.resolve(unit.path()), root_dir))
    file_paths = _find_file_paths(abs_root_dir, "**/screens/*/*/*.png")
    file_dirs = [os.path.dirname(f) for f in file_paths]

    rename_from, rename_to = [os.path.relpath(os.path.normpath(os.path.join(mod_dir, d)), test_for_path)
                              for d in [root_dir, out_dir]]

    test_record.update({
        "TS-TEST-DATA-DIRS": ytest.serialize_list(_resolve_module_files(unit, mod_dir, file_dirs)),
        "TS-TEST-DATA-DIRS-RENAME": "{}:{}".format(rename_from, rename_to),
    })


def _setup_eslint(unit):
    if not _is_tests_enabled(unit):
        return

    if unit.get("_NO_LINT_VALUE") == "none":
        return

    lint_files = ytest.get_values_list(unit, "_TS_LINT_SRCS_VALUE")
    if not lint_files:
        return

    mod_dir = unit.get("MODDIR")
    lint_files = _resolve_module_files(unit, mod_dir, lint_files)
    deps = _create_pm(unit).get_peers_from_package_json()
    test_record = {
        "ESLINT_CONFIG_NAME": unit.get("ESLINT_CONFIG_NAME"),
    }

    _add_test(unit, "eslint", lint_files, deps, test_record, mod_dir)


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


def _add_test(unit, test_type, test_files, deps=None, test_record=None, test_cwd=None):
    from lib.nots.package_manager import constants

    if deps:
        unit.ondepends(deps)

    test_dir = ytest.get_norm_unit_path(unit)
    full_test_record = {
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
        "NODEJS-ROOT-VAR-NAME": unit.get("NODEJS_ROOT_VAR_NAME"),
        "NODE-MODULES-BUNDLE-FILENAME": constants.NODE_MODULES_WORKSPACE_BUNDLE_FILENAME,
        "CUSTOM-DEPENDENCIES": " ".join(deps) if deps else "",
    }

    if test_record:
        full_test_record.update(test_record)

    data = ytest.dump_test(unit, full_test_record)
    if data:
        unit.set_property(["DART_DATA", data])


def _set_nodejs_root(unit):
    pm = _create_pm(unit)

    # example: >= 12.18.4
    version_range = pm.load_package_json_from_dir(pm.sources_path).get_nodejs_version()

    # example: Version(12, 18, 4)
    node_version = _select_matching_node_version(version_range)

    # example: NODEJS_12_18_4_RESOURCE_GLOBAL
    yamake_node_version_var = "NODEJS_{}_RESOURCE_GLOBAL".format(str(node_version).replace(".", "_"))

    unit.set(["NODEJS_ROOT", "${}".format(yamake_node_version_var)])
    unit.set(["NODEJS_BIN", "${}/node".format(yamake_node_version_var)])
    unit.set(["NODEJS_ROOT_VAR_NAME", yamake_node_version_var])


def _select_matching_node_version(range_str):
    """
    :param str range_str:
    :rtype: Version
    """
    from lib.nots.constants import SUPPORTED_NODE_VERSIONS, DEFAULT_NODE_VERSION
    from lib.nots.semver import VersionRange

    if range_str is None:
        return DEFAULT_NODE_VERSION

    try:
        range = VersionRange.from_str(range_str)

        # assuming SUPPORTED_NODE_VERSIONS is sorted from the lowest to highest version
        # we stop the loop as early as possible and hence return the lowest compatible version
        for version in SUPPORTED_NODE_VERSIONS:
            if range.is_satisfied_by(version):
                return version

        raise ValueError("There is no allowed version to satisfy this range: '{}'".format(range_str))
    except Exception as error:
        raise Exception(
            "Requested nodejs version range '{}'' could not be satisfied. Please use a range that would include one of the following: {}.\nFor further details please visit the link: {}\nOriginal error: {}"
            .format(range_str, map(str, SUPPORTED_NODE_VERSIONS), "https://nda.ya.ru/t/ulU4f5Ru5egzHV", str(error))
        )
