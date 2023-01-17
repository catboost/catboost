import os
import re
import sys
import json
import copy
import base64
import shlex
import _common
import lib._metric_resolvers as mr
import _test_const as consts
import _requirements as reqs
import StringIO
import subprocess
import collections

import ymake


MDS_URI_PREFIX = 'https://storage.yandex-team.ru/get-devtools/'
MDS_SCHEME = 'mds'
CANON_DATA_DIR_NAME = 'canondata'
CANON_OUTPUT_STORAGE = 'canondata_storage'
CANON_RESULT_FILE_NAME = 'result.json'
CANON_MDS_RESOURCE_REGEX = re.compile(re.escape(MDS_URI_PREFIX) + r'(.*?)($|#)')
CANON_SB_VAULT_REGEX = re.compile(r"\w+=(value|file):[-\w]+:\w+")
CANON_SBR_RESOURCE_REGEX = re.compile(r'(sbr:/?/?(\d+))')

VALID_NETWORK_REQUIREMENTS = ("full", "restricted")
VALID_DNS_REQUIREMENTS = ("default", "local", "dns64")
BLOCK_SEPARATOR = '============================================================='
SPLIT_FACTOR_MAX_VALUE = 1000
SPLIT_FACTOR_TEST_FILES_MAX_VALUE = 4250
PARTITION_MODS = ('SEQUENTIAL', 'MODULO')
DEFAULT_TIDY_CONFIG = "build/config/tests/clang_tidy/config.yaml"
DEFAULT_TIDY_CONFIG_MAP_PATH = "build/yandex_specific/config/clang_tidy/tidy_default_map.json"
PROJECT_TIDY_CONFIG_MAP_PATH = "build/yandex_specific/config/clang_tidy/tidy_project_map.json"


tidy_config_map = None

def ontest_data(unit, *args):
    ymake.report_configure_error("TEST_DATA is removed in favour of DATA")


def save_in_file(filepath, data):
    if filepath:
        with open(filepath, 'a') as file_handler:
            if os.stat(filepath).st_size == 0:
                print >>file_handler,  BLOCK_SEPARATOR
            print >> file_handler, data


def prepare_recipes(data):
    data = data.replace('"USE_RECIPE_DELIM"', "\n")
    data = data.replace("$TEST_RECIPES_VALUE", "")
    return base64.b64encode(data or "")


def prepare_env(data):
    data = data.replace("$TEST_ENV_VALUE", "")
    return serialize_list(shlex.split(data))


def is_yt_spec_contain_pool_info(filename):  # XXX switch to yson in ymake + perf test for configure
    pool_re = re.compile(r"""['"]*pool['"]*\s*?=""")
    cypress_root_re = re.compile(r"""['"]*cypress_root['"]*\s*=""")
    with open(filename, 'r') as afile:
        yt_spec = afile.read()
        return pool_re.search(yt_spec) and cypress_root_re.search(yt_spec)


def validate_sb_vault(name, value):
    if not CANON_SB_VAULT_REGEX.match(value):
        return "sb_vault value '{}' should follow pattern <ENV_NAME>=:<value|file>:<owner>:<vault key>".format(value)


def validate_numerical_requirement(name, value):
    if mr.resolve_value(value) is None:
        return "Cannot convert [[imp]]{}[[rst]] to the proper [[imp]]{}[[rst]] requirement value".format(value, name)


def validate_choice_requirement(name, val, valid):
    if val not in valid:
        return "Unknown [[imp]]{}[[rst]] requirement: [[imp]]{}[[rst]], choose from [[imp]]{}[[rst]]".format(name, val, ", ".join(valid))


def validate_force_sandbox_requirement(name, value, test_size, is_force_sandbox, in_autocheck, is_fuzzing, is_kvm, is_ytexec_run, check_func):
    if is_force_sandbox or not in_autocheck or is_fuzzing or is_ytexec_run:
        if value == 'all':
            return
        return validate_numerical_requirement(name, value)
    error_msg = validate_numerical_requirement(name, value)
    if error_msg:
        return error_msg
    return check_func(mr.resolve_value(value), test_size, is_kvm)


# TODO: Remove is_kvm param when there will be guarantees on RAM
def validate_requirement(req_name, value, test_size, is_force_sandbox, in_autocheck, is_fuzzing, is_kvm, is_ytexec_run):
    req_checks = {
        'container': validate_numerical_requirement,
        'cpu': lambda n, v: validate_force_sandbox_requirement(n, v, test_size, is_force_sandbox, in_autocheck, is_fuzzing, is_kvm, is_ytexec_run, reqs.check_cpu),
        'disk_usage': validate_numerical_requirement,
        'dns': lambda n, v: validate_choice_requirement(n, v, VALID_DNS_REQUIREMENTS),
        'kvm': None,
        'network': lambda n, v: validate_choice_requirement(n, v, VALID_NETWORK_REQUIREMENTS),
        'ram': lambda n, v: validate_force_sandbox_requirement(n, v, test_size, is_force_sandbox, in_autocheck, is_fuzzing, is_kvm, is_ytexec_run, reqs.check_ram),
        'ram_disk': lambda n, v: validate_force_sandbox_requirement(n, v, test_size, is_force_sandbox, in_autocheck, is_fuzzing, is_kvm, is_ytexec_run, reqs.check_ram_disk),
        'sb': None,
        'sb_vault': validate_sb_vault,
    }

    if req_name not in req_checks:
        return "Unknown requirement: [[imp]]{}[[rst]], choose from [[imp]]{}[[rst]]".format(req_name, ", ".join(sorted(req_checks)))

    if req_name in ('container', 'disk') and not is_force_sandbox:
        return "Only [[imp]]LARGE[[rst]] tests without [[imp]]ya:force_distbuild[[rst]] tag can have [[imp]]{}[[rst]] requirement".format(req_name)

    check_func = req_checks[req_name]
    if check_func:
        return check_func(req_name, value)


def validate_test(unit, kw):
    def get_list(key):
        return deserialize_list(kw.get(key, ""))

    valid_kw = copy.deepcopy(kw)
    errors = []
    warnings = []

    if valid_kw.get('SCRIPT-REL-PATH') == 'boost.test':
        project_path = valid_kw.get('BUILD-FOLDER-PATH', "")
        if not project_path.startswith(("contrib", "mail", "maps", "tools/idl", "metrika", "devtools", "mds", "yandex_io", "smart_devices")):
            errors.append("BOOSTTEST is not allowed here")
    elif valid_kw.get('SCRIPT-REL-PATH') == 'gtest':
        project_path = valid_kw.get('BUILD-FOLDER-PATH', "")
        if not project_path.startswith(("contrib", "devtools", "mail", "mds")):
            errors.append("GTEST_UGLY is not allowed here, use GTEST instead")

    size_timeout = collections.OrderedDict(sorted(consts.TestSize.DefaultTimeouts.items(), key=lambda t: t[1]))

    size = valid_kw.get('SIZE', consts.TestSize.Small).lower()
    # TODO: use set instead list
    tags = get_list("TAG")
    requirements_orig = get_list("REQUIREMENTS")
    in_autocheck = "ya:not_autocheck" not in tags and 'ya:manual' not in tags
    is_fat = 'ya:fat' in tags
    is_force_sandbox = 'ya:force_distbuild' not in tags and is_fat
    is_ytexec_run = 'ya:yt' in tags
    is_fuzzing = valid_kw.get("FUZZING", False)
    is_kvm = 'kvm' in requirements_orig
    requirements = {}
    list_requirements = ('sb_vault')
    for req in requirements_orig:
        if req in ('kvm', ):
            requirements[req] = str(True)
            continue

        if ":" in req:
            req_name, req_value = req.split(":", 1)
            if req_name in list_requirements:
                requirements[req_name] = ",".join(filter(None, [requirements.get(req_name), req_value]))
            else:
                if req_name in requirements:
                    if req_value in ["0"]:
                        warnings.append("Requirement [[imp]]{}[[rst]] is dropped [[imp]]{}[[rst]] -> [[imp]]{}[[rst]]".format(req_name, requirements[req_name], req_value))
                        del requirements[req_name]
                    elif requirements[req_name] != req_value:
                        warnings.append("Requirement [[imp]]{}[[rst]] is redefined [[imp]]{}[[rst]] -> [[imp]]{}[[rst]]".format(req_name, requirements[req_name], req_value))
                        requirements[req_name] = req_value
                else:
                    requirements[req_name] = req_value
        else:
            errors.append("Invalid requirement syntax [[imp]]{}[[rst]]: expect <requirement>:<value>".format(req))

    if not errors:
        for req_name, req_value in requirements.items():
            error_msg = validate_requirement(req_name, req_value, size, is_force_sandbox, in_autocheck, is_fuzzing, is_kvm, is_ytexec_run)
            if error_msg:
                errors += [error_msg]

    invalid_requirements_for_distbuild = [requirement for requirement in requirements.keys() if requirement not in ('ram', 'ram_disk', 'cpu', 'network')]
    sb_tags = [tag for tag in tags if tag.startswith('sb:')]

    if is_fat:
        if size != consts.TestSize.Large:
            errors.append("Only LARGE test may have ya:fat tag")

        if in_autocheck and not is_force_sandbox:
            if invalid_requirements_for_distbuild:
                errors.append("'{}' REQUIREMENTS options can be used only for FAT tests without ya:force_distbuild tag. Remove TAG(ya:force_distbuild) or an option.".format(invalid_requirements_for_distbuild))
            if sb_tags:
                errors.append("You can set sandbox tags '{}' only for FAT tests without ya:force_distbuild. Remove TAG(ya:force_sandbox) or sandbox tags.".format(sb_tags))
            if 'ya:sandbox_coverage' in tags:
                errors.append("You can set 'ya:sandbox_coverage' tag only for FAT tests without ya:force_distbuild.")
            if is_ytexec_run:
                errors.append("Running LARGE tests over YT (ya:yt) on Distbuild (ya:force_distbuild) is forbidden. Consider removing TAG(ya:force_distbuild).")
    else:
        if is_force_sandbox:
            errors.append('ya:force_sandbox can be used with LARGE tests only')
        if 'ya:nofuse' in tags:
            errors.append('ya:nofuse can be used with LARGE tests only')
        if 'ya:privileged' in tags:
            errors.append("ya:privileged can be used with LARGE tests only")
        if in_autocheck and size == consts.TestSize.Large:
            errors.append("LARGE test must have ya:fat tag")

    if 'ya:privileged' in tags and 'container' not in requirements:
        errors.append("Only tests with 'container' requirement can have 'ya:privileged' tag")

    if size not in size_timeout:
        errors.append("Unknown test size: [[imp]]{}[[rst]], choose from [[imp]]{}[[rst]]".format(size.upper(), ", ".join([sz.upper() for sz in size_timeout.keys()])))
    else:
        try:
            timeout = int(valid_kw.get('TEST-TIMEOUT', size_timeout[size]) or size_timeout[size])
            script_rel_path = valid_kw.get('SCRIPT-REL-PATH')
            if timeout < 0:
                raise Exception("Timeout must be > 0")
            if size_timeout[size] < timeout and in_autocheck and script_rel_path != 'java.style':
                suggested_size = None
                for s, t in size_timeout.items():
                    if timeout <= t:
                        suggested_size = s
                        break

                if suggested_size:
                    suggested_size = ", suggested size: [[imp]]{}[[rst]]".format(suggested_size.upper())
                else:
                    suggested_size = ""
                errors.append("Max allowed timeout for test size [[imp]]{}[[rst]] is [[imp]]{} sec[[rst]]{}".format(size.upper(), size_timeout[size], suggested_size))
        except Exception as e:
            errors.append("Error when parsing test timeout: [[bad]]{}[[rst]]".format(e))

        requirements_list = []
        for req_name, req_value in requirements.iteritems():
            requirements_list.append(req_name + ":" + req_value)
        valid_kw['REQUIREMENTS'] = serialize_list(requirements_list)

    if valid_kw.get("FUZZ-OPTS"):
        for option in get_list("FUZZ-OPTS"):
            if not option.startswith("-"):
                errors.append("Unrecognized fuzzer option '[[imp]]{}[[rst]]'. All fuzzer options should start with '-'".format(option))
                break
            eqpos = option.find("=")
            if eqpos == -1 or len(option) == eqpos + 1:
                errors.append("Unrecognized fuzzer option '[[imp]]{}[[rst]]'. All fuzzer options should obtain value specified after '='".format(option))
                break
            if option[eqpos - 1] == " " or option[eqpos + 1] == " ":
                errors.append("Spaces are not allowed: '[[imp]]{}[[rst]]'".format(option))
                break
            if option[:eqpos] in ("-runs", "-dict", "-jobs", "-workers", "-artifact_prefix", "-print_final_stats"):
                errors.append("You can't use '[[imp]]{}[[rst]]' - it will be automatically calculated or configured during run".format(option))
                break

    if valid_kw.get("YT-SPEC"):
        if not is_ytexec_run:
            errors.append("You can use YT_SPEC macro only tests marked with ya:yt tag")
        else:
            for filename in get_list("YT-SPEC"):
                filename = unit.resolve('$S/' + filename)
                if not os.path.exists(filename):
                    errors.append("File '{}' specified in the YT_SPEC macro doesn't exist".format(filename))
                    continue
                if is_yt_spec_contain_pool_info(filename) and "ya:external" not in tags:
                    tags.append("ya:external")
                    tags.append("ya:yt_research_pool")

    if valid_kw.get("USE_ARCADIA_PYTHON") == "yes" and valid_kw.get("SCRIPT-REL-PATH") == "py.test":
        errors.append("PYTEST_SCRIPT is deprecated")

    partition = valid_kw.get('TEST_PARTITION', 'SEQUENTIAL')
    if partition not in PARTITION_MODS:
        raise ValueError('partition mode should be one of {}, detected: {}'.format(PARTITION_MODS, partition))

    if valid_kw.get('SPLIT-FACTOR'):
        if valid_kw.get('FORK-MODE') == 'none':
            errors.append('SPLIT_FACTOR must be use with FORK_TESTS() or FORK_SUBTESTS() macro')

        value = 1
        try:
            value = int(valid_kw.get('SPLIT-FACTOR'))
            if value <= 0:
                raise ValueError("must be > 0")
            if value > SPLIT_FACTOR_MAX_VALUE:
                raise ValueError("the maximum allowed value is {}".format(SPLIT_FACTOR_MAX_VALUE))
        except ValueError as e:
            errors.append('Incorrect SPLIT_FACTOR value: {}'.format(e))

        if valid_kw.get('FORK-TEST-FILES') and size != consts.TestSize.Large:
            nfiles = count_entries(valid_kw.get('TEST-FILES'))
            if nfiles * value > SPLIT_FACTOR_TEST_FILES_MAX_VALUE:
                errors.append('Too much chunks generated:{} (limit: {}). Remove FORK_TEST_FILES() macro or reduce SPLIT_FACTOR({}).'.format(
                    nfiles * value, SPLIT_FACTOR_TEST_FILES_MAX_VALUE, value))

    unit_path = get_norm_unit_path(unit)
    if not is_fat and "ya:noretries" in tags and not is_ytexec_run \
            and not unit_path.startswith("devtools/dummy_arcadia/test/noretries"):
        errors.append("Only LARGE tests can have 'ya:noretries' tag")

    if errors:
        return None, warnings, errors

    return valid_kw, warnings, errors


def get_norm_unit_path(unit, extra=None):
    path = _common.strip_roots(unit.path())
    if extra:
        return '{}/{}'.format(path, extra)
    return path


def dump_test(unit, kw):
    valid_kw, warnings, errors = validate_test(unit, kw)
    for w in warnings:
        unit.message(['warn', w])
    for e in errors:
        ymake.report_configure_error(e)
    if valid_kw is None:
        return None
    string_handler = StringIO.StringIO()
    for k, v in valid_kw.iteritems():
        print >>string_handler, k + ': ' + v
    print >>string_handler, BLOCK_SEPARATOR
    data = string_handler.getvalue()
    string_handler.close()
    return data


def serialize_list(lst):
    lst = filter(None, lst)
    return '\"' + ';'.join(lst) + '\"' if lst else ''


def deserialize_list(val):
    return filter(None, val.replace('"', "").split(";"))


def count_entries(x):
    # see (de)serialize_list
    assert x is None or isinstance(x, str), type(x)
    if not x:
        return 0
    return x.count(";") + 1


def get_values_list(unit, key):
    res = map(str.strip, (unit.get(key) or '').replace('$' + key, '').strip().split())
    return [r for r in res if r and r not in ['""', "''"]]


def get_norm_paths(unit, key):
    # return paths without trailing (back)slash
    return [x.rstrip('\\/') for x in get_values_list(unit, key)]


def get_unit_list_variable(unit, name):
    items = unit.get(name)
    if items:
        items = items.split(' ')
        assert items[0] == "${}".format(name), (items, name)
        return items[1:]
    return []


def implies(a, b):
    return bool((not a) or b)


def match_coverage_extractor_requirements(unit):
    # we shouldn't add test if
    return all([
        # tests are not requested
        unit.get("TESTS_REQUESTED") == "yes",
        # build doesn't imply clang coverage, which supports segment extraction from the binaries
        unit.get("CLANG_COVERAGE") == "yes",
        # contrib wasn't requested
        implies(get_norm_unit_path(unit).startswith("contrib/"), unit.get("ENABLE_CONTRIB_COVERAGE") == "yes"),
    ])


def get_tidy_config_map(unit, map_path):
    config_map_path = unit.resolve(os.path.join("$S", map_path))
    config_map = {}
    try:
        with open(config_map_path, 'r') as afile:
            config_map = json.load(afile)
    except ValueError:
        ymake.report_configure_error("{} is invalid json".format(map_path))
    except Exception as e:
        ymake.report_configure_error(str(e))
    return config_map


def get_default_tidy_config(unit):
    unit_path = get_norm_unit_path(unit)
    tidy_default_config_map = get_tidy_config_map(unit, DEFAULT_TIDY_CONFIG_MAP_PATH)
    for project_prefix, config_path in tidy_default_config_map.items():
        if unit_path.startswith(project_prefix):
            return config_path
    return DEFAULT_TIDY_CONFIG


ordered_tidy_map = None


def get_project_tidy_config(unit):
    global ordered_tidy_map
    if ordered_tidy_map is None:
        ordered_tidy_map = list(reversed(sorted(get_tidy_config_map(unit, PROJECT_TIDY_CONFIG_MAP_PATH).items())))
    unit_path = get_norm_unit_path(unit)

    for project_prefix, config_path in ordered_tidy_map:
        if unit_path.startswith(project_prefix):
            return config_path
    else:
        return get_default_tidy_config(unit)


def onadd_ytest(unit, *args):
    keywords = {"DEPENDS": -1, "DATA": -1, "TIMEOUT": 1, "FORK_MODE": 1, "SPLIT_FACTOR": 1,
                "FORK_SUBTESTS": 0, "FORK_TESTS": 0}
    flat_args, spec_args = _common.sort_by_keywords(keywords, args)

    test_data = sorted(_common.filter_out_by_keyword(spec_args.get('DATA', []) + get_norm_paths(unit, 'TEST_DATA_VALUE'), 'AUTOUPDATED'))

    if flat_args[1] == "fuzz.test":
        unit.ondata("arcadia/fuzzing/{}/corpus.json".format(get_norm_unit_path(unit)))
    elif flat_args[1] == "go.test":
        data, _ = get_canonical_test_resources(unit)
        test_data += data
    elif flat_args[1] == "coverage.extractor" and not match_coverage_extractor_requirements(unit):
        # XXX
        # Current ymake implementation doesn't allow to call macro inside the 'when' body
        # that's why we add ADD_YTEST(coverage.extractor) to every PROGRAM entry and check requirements later
        return
    elif flat_args[1] == "clang_tidy" and unit.get("TIDY_ENABLED") != "yes":
        # Graph is not prepared
        return
    elif flat_args[1] == "no.test":
        return
    test_size = ''.join(spec_args.get('SIZE', [])) or unit.get('TEST_SIZE_NAME') or ''
    test_tags = serialize_list(_get_test_tags(unit, spec_args))
    test_timeout = ''.join(spec_args.get('TIMEOUT', [])) or unit.get('TEST_TIMEOUT') or ''
    test_requirements = spec_args.get('REQUIREMENTS', []) + get_values_list(unit, 'TEST_REQUIREMENTS_VALUE')

    if flat_args[1] != "clang_tidy" and unit.get("TIDY_ENABLED") == "yes":
        # graph changed for clang_tidy tests
        if flat_args[1] in ("unittest.py", "gunittest", "g_benchmark"):
            flat_args[1] = "clang_tidy"
            test_size = 'SMALL'
            test_tags = ''
            test_timeout = "60"
            test_requirements = []
            unit.set(["TEST_YT_SPEC_VALUE", ""])
        else:
            return

    if flat_args[1] == "clang_tidy" and unit.get("TIDY_ENABLED") == "yes":
        if unit.get("TIDY_CONFIG"):
            default_config_path = unit.get("TIDY_CONFIG")
            project_config_path = unit.get("TIDY_CONFIG")
        else:
            default_config_path = get_default_tidy_config(unit)
            project_config_path = get_project_tidy_config(unit)

        unit.set(["DEFAULT_TIDY_CONFIG", default_config_path])
        unit.set(["PROJECT_TIDY_CONFIG", project_config_path])

    fork_mode = []
    if 'FORK_SUBTESTS' in spec_args:
        fork_mode.append('subtests')
    if 'FORK_TESTS' in spec_args:
        fork_mode.append('tests')
    fork_mode = fork_mode or spec_args.get('FORK_MODE', []) or unit.get('TEST_FORK_MODE').split()
    fork_mode = ' '.join(fork_mode) if fork_mode else ''

    unit_path = get_norm_unit_path(unit)

    test_record = {
        'TEST-NAME': flat_args[0],
        'SCRIPT-REL-PATH': flat_args[1],
        'TESTED-PROJECT-NAME': unit.name(),
        'TESTED-PROJECT-FILENAME': unit.filename(),
        'SOURCE-FOLDER-PATH': unit_path,
        # TODO get rid of BUILD-FOLDER-PATH
        'BUILD-FOLDER-PATH': unit_path,
        'BINARY-PATH': "{}/{}".format(unit_path, unit.filename()),
        'GLOBAL-LIBRARY-PATH': unit.global_filename(),
        'CUSTOM-DEPENDENCIES': ' '.join(spec_args.get('DEPENDS', []) + get_values_list(unit, 'TEST_DEPENDS_VALUE')),
        'TEST-RECIPES': prepare_recipes(unit.get("TEST_RECIPES_VALUE")),
        'TEST-ENV': prepare_env(unit.get("TEST_ENV_VALUE")),
        #  'TEST-PRESERVE-ENV': 'da',
        'TEST-DATA': serialize_list(test_data),
        'TEST-TIMEOUT': test_timeout,
        'FORK-MODE': fork_mode,
        'SPLIT-FACTOR': ''.join(spec_args.get('SPLIT_FACTOR', [])) or unit.get('TEST_SPLIT_FACTOR') or '',
        'SIZE': test_size,
        'TAG': test_tags,
        'REQUIREMENTS': serialize_list(test_requirements),
        'TEST-CWD': unit.get('TEST_CWD_VALUE') or '',
        'FUZZ-DICTS': serialize_list(spec_args.get('FUZZ_DICTS', []) + get_unit_list_variable(unit, 'FUZZ_DICTS_VALUE')),
        'FUZZ-OPTS': serialize_list(spec_args.get('FUZZ_OPTS', []) + get_unit_list_variable(unit, 'FUZZ_OPTS_VALUE')),
        'YT-SPEC': serialize_list(spec_args.get('YT_SPEC', []) + get_unit_list_variable(unit, 'TEST_YT_SPEC_VALUE')),
        'BLOB': unit.get('TEST_BLOB_DATA') or '',
        'SKIP_TEST': unit.get('SKIP_TEST_VALUE') or '',
        'TEST_IOS_DEVICE_TYPE': unit.get('TEST_IOS_DEVICE_TYPE_VALUE') or '',
        'TEST_IOS_RUNTIME_TYPE': unit.get('TEST_IOS_RUNTIME_TYPE_VALUE') or '',
        'ANDROID_APK_TEST_ACTIVITY': unit.get('ANDROID_APK_TEST_ACTIVITY_VALUE') or '',
        'TEST_PARTITION': unit.get("TEST_PARTITION") or 'SEQUENTIAL',
        'GO_BENCH_TIMEOUT': unit.get('GO_BENCH_TIMEOUT') or '',
    }

    if flat_args[1] == "go.bench":
        if "ya:run_go_benchmark" not in test_record["TAG"]:
            return
        else:
            test_record["TEST-NAME"] += "_bench"

    if flat_args[1] == 'fuzz.test' and unit.get('FUZZING') == 'yes':
        test_record['FUZZING'] = '1'
        # use all cores if fuzzing requested
        test_record['REQUIREMENTS'] = serialize_list(filter(None, deserialize_list(test_record['REQUIREMENTS']) + ["cpu:all", "ram:all"]))

    data = dump_test(unit, test_record)
    if data:
        unit.set_property(["DART_DATA", data])
        save_in_file(unit.get('TEST_DART_OUT_FILE'), data)


def java_srcdirs_to_data(unit, var):
    extra_data = []
    for srcdir in (unit.get(var) or '').replace('$' + var, '').split():
        if srcdir == '.':
            srcdir = unit.get('MODDIR')
        if srcdir.startswith('${ARCADIA_ROOT}/') or srcdir.startswith('$ARCADIA_ROOT/'):
            srcdir = srcdir.replace('${ARCADIA_ROOT}/', '$S/')
            srcdir = srcdir.replace('$ARCADIA_ROOT/', '$S/')
        if srcdir.startswith('${CURDIR}/') or srcdir.startswith('$CURDIR/'):
            srcdir = srcdir.replace('${CURDIR}/', os.path.join('$S', unit.get('MODDIR')))
            srcdir = srcdir.replace('$CURDIR/', os.path.join('$S', unit.get('MODDIR')))
        srcdir = unit.resolve_arc_path(srcdir)
        if not srcdir.startswith('$'):
            srcdir = os.path.join('$S', unit.get('MODDIR'), srcdir)
        if srcdir.startswith('$S'):
            extra_data.append(srcdir.replace('$S', 'arcadia'))
    return serialize_list(extra_data)


def onadd_check(unit, *args):
    if unit.get("TIDY") == "yes":
        # graph changed for clang_tidy tests
        return
    flat_args, spec_args = _common.sort_by_keywords({"DEPENDS": -1, "TIMEOUT": 1, "DATA": -1, "TAG": -1,
                                                     "REQUIREMENTS": -1, "FORK_MODE": 1, "SPLIT_FACTOR": 1,
                                                     "FORK_SUBTESTS": 0, "FORK_TESTS": 0, "SIZE": 1}, args)
    check_type = flat_args[0]

    if check_type in ("check.data", "check.resource") and unit.get('VALIDATE_DATA') == "no":
        return

    test_dir = get_norm_unit_path(unit)

    test_timeout = ''
    fork_mode = ''
    extra_test_data = ''
    extra_test_dart_data = {}
    ymake_java_test = unit.get('YMAKE_JAVA_TEST') == 'yes'
    use_arcadia_python = unit.get('USE_ARCADIA_PYTHON')
    uid_ext = ''
    script_rel_path = check_type
    test_files = flat_args[1:]

    supported_no_lint_values = ('none', 'none_internal', 'ktlint')
    no_lint_value = unit.get('_NO_LINT_VALUE')
    if no_lint_value and no_lint_value not in supported_no_lint_values:
        ymake.report_configure_error('Unsupported value for NO_LINT macro: {}'.format(no_lint_value))

    if check_type in ["check.data", "check.resource"]:
        uid_ext = unit.get("SBR_UID_EXT").split(" ", 1)[-1]  # strip variable name

    if check_type in ["flake8.py2", "flake8.py3", "black"]:
        fork_mode = unit.get('TEST_FORK_MODE') or ''
    elif check_type == "JAVA_STYLE":
        if ymake_java_test and not unit.get('ALL_SRCDIRS') or '':
            return
        if len(flat_args) < 2:
            raise Exception("Not enough arguments for JAVA_STYLE check")
        check_level = flat_args[1]
        allowed_levels = {
            'base': '/yandex_checks.xml',
            'strict': '/yandex_checks_strict.xml',
            'extended': '/yandex_checks_extended.xml',
            'library': '/yandex_checks_library.xml',
        }
        if check_level not in allowed_levels:
            raise Exception("'{}' is not allowed in LINT(), use one of {}".format(check_level, allowed_levels.keys()))
        test_files[0] = allowed_levels[check_level]  # replace check_level with path to config file
        script_rel_path = "java.style"
        test_timeout = '240'
        fork_mode = unit.get('TEST_FORK_MODE') or ''
        if ymake_java_test:
            extra_test_data = java_srcdirs_to_data(unit, 'ALL_SRCDIRS')

        # jstyle should use the latest jdk
        unit.onpeerdir([unit.get('JDK_LATEST_PEERDIR')])
        extra_test_dart_data['JDK_LATEST_VERSION'] = unit.get('JDK_LATEST_VERSION')
        # TODO remove when ya-bin will be released (https://st.yandex-team.ru/DEVTOOLS-9611)
        extra_test_dart_data['JDK_RESOURCE'] = 'JDK' + (unit.get('JDK_VERSION') or unit.get('JDK_REAL_VERSION') or '_DEFAULT')
    elif check_type == "gofmt":
        if test_files:
            test_dir = os.path.dirname(test_files[0]).lstrip("$S/")
    elif check_type == "check.data":
        data_re = re.compile(r"sbr:/?/?(\d+)=?.*")
        data = flat_args[1:]
        resources = []
        for f in data:
            matched = re.match(data_re, f)
            if matched:
                resources.append(matched.group(1))
        if resources:
            test_files = resources
        else:
            return

    serialized_test_files = serialize_list(test_files)

    test_record = {
        'TEST-NAME': check_type.lower(),
        'TEST-TIMEOUT': test_timeout,
        'SCRIPT-REL-PATH': script_rel_path,
        'TESTED-PROJECT-NAME': os.path.basename(test_dir),
        'SOURCE-FOLDER-PATH': test_dir,
        'CUSTOM-DEPENDENCIES': " ".join(spec_args.get('DEPENDS', [])),
        'TEST-DATA': extra_test_data,
        'TEST-ENV': prepare_env(unit.get("TEST_ENV_VALUE")),
        'SBR-UID-EXT': uid_ext,
        'SPLIT-FACTOR': '',
        'TEST_PARTITION': 'SEQUENTIAL',
        'FORK-MODE': fork_mode,
        'FORK-TEST-FILES': '',
        'SIZE': 'SMALL',
        'TAG': '',
        'REQUIREMENTS': '',
        'USE_ARCADIA_PYTHON': use_arcadia_python or '',
        'OLD_PYTEST': 'no',
        'PYTHON-PATHS': '',
        # TODO remove FILES, see DEVTOOLS-7052
        'FILES': serialized_test_files,
        'TEST-FILES': serialized_test_files,
        'NO_JBUILD': 'yes' if ymake_java_test else 'no',
    }
    test_record.update(extra_test_dart_data)

    data = dump_test(unit, test_record)
    if data:
        unit.set_property(["DART_DATA", data])
        save_in_file(unit.get('TEST_DART_OUT_FILE'), data)


def on_register_no_check_imports(unit):
    s = unit.get('NO_CHECK_IMPORTS_FOR_VALUE')
    if s not in ('', 'None'):
        unit.onresource(['-', 'py/no_check_imports/{}="{}"'.format(_common.pathid(s), s)])


def onadd_check_py_imports(unit, *args):
    if unit.get("TIDY") == "yes":
        # graph changed for clang_tidy tests
        return
    if unit.get('NO_CHECK_IMPORTS_FOR_VALUE').strip() == "":
        return
    unit.onpeerdir(['library/python/testing/import_test'])
    check_type = "py.imports"
    test_dir = get_norm_unit_path(unit)

    use_arcadia_python = unit.get('USE_ARCADIA_PYTHON')
    test_files = serialize_list([get_norm_unit_path(unit, unit.filename())])
    test_record = {
        'TEST-NAME': "pyimports",
        'TEST-TIMEOUT': '',
        'SCRIPT-REL-PATH': check_type,
        'TESTED-PROJECT-NAME': os.path.basename(test_dir),
        'SOURCE-FOLDER-PATH': test_dir,
        'CUSTOM-DEPENDENCIES': '',
        'TEST-DATA': '',
        'TEST-ENV': prepare_env(unit.get("TEST_ENV_VALUE")),
        'SPLIT-FACTOR': '',
        'TEST_PARTITION': 'SEQUENTIAL',
        'FORK-MODE': '',
        'FORK-TEST-FILES': '',
        'SIZE': 'SMALL',
        'TAG': '',
        'USE_ARCADIA_PYTHON': use_arcadia_python or '',
        'OLD_PYTEST': 'no',
        'PYTHON-PATHS': '',
        # TODO remove FILES, see DEVTOOLS-7052
        'FILES': test_files,
        'TEST-FILES': test_files,
    }
    if unit.get('NO_CHECK_IMPORTS_FOR_VALUE') != "None":
        test_record["NO-CHECK"] = serialize_list(get_values_list(unit, 'NO_CHECK_IMPORTS_FOR_VALUE') or ["*"])
    else:
        test_record["NO-CHECK"] = ''
    data = dump_test(unit, test_record)
    if data:
        unit.set_property(["DART_DATA", data])
        save_in_file(unit.get('TEST_DART_OUT_FILE'), data)


def onadd_pytest_script(unit, *args):
    if unit.get("TIDY") == "yes":
        # graph changed for clang_tidy tests
        return
    unit.set(["PYTEST_BIN", "no"])
    custom_deps = get_values_list(unit, 'TEST_DEPENDS_VALUE')
    timeout = filter(None, [unit.get(["TEST_TIMEOUT"])])

    if timeout:
        timeout = timeout[0]
    else:
        timeout = '0'
    test_type = args[0]
    fork_mode = unit.get('TEST_FORK_MODE').split() or ''
    split_factor = unit.get('TEST_SPLIT_FACTOR') or ''
    test_size = unit.get('TEST_SIZE_NAME') or ''

    test_files = get_values_list(unit, 'TEST_SRCS_VALUE')
    tags = _get_test_tags(unit)
    requirements = get_values_list(unit, 'TEST_REQUIREMENTS_VALUE')
    test_data = get_norm_paths(unit, 'TEST_DATA_VALUE')
    data, data_files = get_canonical_test_resources(unit)
    test_data += data
    python_paths = get_values_list(unit, 'TEST_PYTHON_PATH_VALUE')
    binary_path = None
    test_cwd = unit.get('TEST_CWD_VALUE') or ''
    _dump_test(unit, test_type, test_files, timeout, get_norm_unit_path(unit), custom_deps, test_data, python_paths, split_factor, fork_mode, test_size, tags, requirements, binary_path, test_cwd=test_cwd, data_files=data_files)


def onadd_pytest_bin(unit, *args):
    if unit.get("TIDY") == "yes":
        # graph changed for clang_tidy tests
        return
    flat, kws = _common.sort_by_keywords({'RUNNER_BIN': 1}, args)
    if flat:
        ymake.report_configure_error(
            'Unknown arguments found while processing add_pytest_bin macro: {!r}'
            .format(flat)
        )

    runner_bin = kws.get('RUNNER_BIN', [None])[0]
    test_type = 'py3test.bin' if (unit.get("PYTHON3") == 'yes') else "pytest.bin"

    add_test_to_dart(unit, test_type, runner_bin=runner_bin)


def add_test_to_dart(unit, test_type, binary_path=None, runner_bin=None):
    if unit.get("TIDY") == "yes":
        # graph changed for clang_tidy tests
        return
    custom_deps = get_values_list(unit, 'TEST_DEPENDS_VALUE')
    timeout = filter(None, [unit.get(["TEST_TIMEOUT"])])
    if timeout:
        timeout = timeout[0]
    else:
        timeout = '0'
    fork_mode = unit.get('TEST_FORK_MODE').split() or ''
    split_factor = unit.get('TEST_SPLIT_FACTOR') or ''
    test_size = unit.get('TEST_SIZE_NAME') or ''
    test_cwd = unit.get('TEST_CWD_VALUE') or ''

    unit_path = unit.path()
    test_files = get_values_list(unit, 'TEST_SRCS_VALUE')
    tags = _get_test_tags(unit)
    requirements = get_values_list(unit, 'TEST_REQUIREMENTS_VALUE')
    test_data = get_norm_paths(unit, 'TEST_DATA_VALUE')
    data, data_files = get_canonical_test_resources(unit)
    test_data += data
    python_paths = get_values_list(unit, 'TEST_PYTHON_PATH_VALUE')
    yt_spec = get_values_list(unit, 'TEST_YT_SPEC_VALUE')
    if not binary_path:
        binary_path = os.path.join(unit_path, unit.filename())
    _dump_test(unit, test_type, test_files, timeout, get_norm_unit_path(unit), custom_deps, test_data, python_paths, split_factor, fork_mode, test_size, tags, requirements, binary_path, test_cwd=test_cwd, runner_bin=runner_bin, yt_spec=yt_spec, data_files=data_files)


def extract_java_system_properties(unit, args):
    if len(args) % 2:
        return [], 'Wrong use of SYSTEM_PROPERTIES in {}: odd number of arguments'.format(unit.path())

    props = []
    for x, y in zip(args[::2], args[1::2]):
        if x == 'FILE':
            if y.startswith('${BINDIR}') or y.startswith('${ARCADIA_BUILD_ROOT}') or y.startswith('/'):
                return [], 'Wrong use of SYSTEM_PROPERTIES in {}: absolute/build file path {}'.format(unit.path(), y)

            y = _common.rootrel_arc_src(y, unit)
            if not os.path.exists(unit.resolve('$S/' + y)):
                return [], 'Wrong use of SYSTEM_PROPERTIES in {}: can\'t resolve {}'.format(unit.path(), y)

            y = '${ARCADIA_ROOT}/' + y
            props.append({'type': 'file', 'path': y})
        else:
            props.append({'type': 'inline', 'key': x, 'value': y})

    return props, None


def onjava_test(unit, *args):
    if unit.get("TIDY") == "yes":
        # graph changed for clang_tidy tests
        return

    assert unit.get('MODULE_TYPE') is not None

    if unit.get('MODULE_TYPE') == 'JTEST_FOR':
        if not unit.get('UNITTEST_DIR'):
            ymake.report_configure_error('skip JTEST_FOR in {}: no args provided'.format(unit.path()))
            return

    java_cp_arg_type = unit.get('JAVA_CLASSPATH_CMD_TYPE_VALUE') or 'MANIFEST'
    if java_cp_arg_type not in ('MANIFEST', 'COMMAND_FILE', 'LIST'):
        ymake.report_configure_error('{}: TEST_JAVA_CLASSPATH_CMD_TYPE({}) are invalid. Choose argument from MANIFEST, COMMAND_FILE or LIST)'.format(unit.path(), java_cp_arg_type))
        return

    unit_path = unit.path()
    path = _common.strip_roots(unit_path)

    test_data = get_norm_paths(unit, 'TEST_DATA_VALUE')
    test_data.append('arcadia/build/scripts/run_junit.py')
    test_data.append('arcadia/build/scripts/unpacking_jtest_runner.py')

    data, data_files = get_canonical_test_resources(unit)
    test_data += data

    props, error_mgs = extract_java_system_properties(unit, get_values_list(unit, 'SYSTEM_PROPERTIES_VALUE'))
    if error_mgs:
        ymake.report_configure_error(error_mgs)
        return
    for prop in props:
        if prop['type'] == 'file':
            test_data.append(prop['path'].replace('${ARCADIA_ROOT}', 'arcadia'))

    props = base64.b64encode(json.dumps(props, encoding='utf-8'))

    test_cwd = unit.get('TEST_CWD_VALUE') or ''  # TODO: validate test_cwd value

    if unit.get('MODULE_TYPE') == 'JUNIT5':
        script_rel_path = 'junit5.test'
    else:
        script_rel_path = 'junit.test'

    ymake_java_test = unit.get('YMAKE_JAVA_TEST') == 'yes'
    test_record = {
        'SOURCE-FOLDER-PATH': path,
        'TEST-NAME': '-'.join([os.path.basename(os.path.dirname(path)), os.path.basename(path)]),
        'SCRIPT-REL-PATH': script_rel_path,
        'TEST-TIMEOUT': unit.get('TEST_TIMEOUT') or '',
        'TESTED-PROJECT-NAME': path,
        'TEST-ENV': prepare_env(unit.get("TEST_ENV_VALUE")),
        #  'TEST-PRESERVE-ENV': 'da',
        'TEST-DATA': serialize_list(sorted(_common.filter_out_by_keyword(test_data, 'AUTOUPDATED'))),
        'FORK-MODE': unit.get('TEST_FORK_MODE') or '',
        'SPLIT-FACTOR': unit.get('TEST_SPLIT_FACTOR') or '',
        'CUSTOM-DEPENDENCIES': ' '.join(get_values_list(unit, 'TEST_DEPENDS_VALUE')),
        'TAG': serialize_list(_get_test_tags(unit)),
        'SIZE': unit.get('TEST_SIZE_NAME') or '',
        'REQUIREMENTS': serialize_list(get_values_list(unit, 'TEST_REQUIREMENTS_VALUE')),
        'TEST-RECIPES': prepare_recipes(unit.get("TEST_RECIPES_VALUE")),

        # JTEST/JTEST_FOR only
        'MODULE_TYPE': unit.get('MODULE_TYPE'),
        'UNITTEST_DIR': unit.get('UNITTEST_DIR') or '',
        'JVM_ARGS': serialize_list(get_values_list(unit, 'JVM_ARGS_VALUE')),
        'SYSTEM_PROPERTIES': props,
        'TEST-CWD': test_cwd,
        'SKIP_TEST': unit.get('SKIP_TEST_VALUE') or '',
        'JAVA_CLASSPATH_CMD_TYPE': java_cp_arg_type,
        'NO_JBUILD': 'yes' if ymake_java_test else 'no',
        'JDK_RESOURCE': 'JDK' + (unit.get('JDK_VERSION') or unit.get('JDK_REAL_VERSION') or '_DEFAULT'),
        'JDK_FOR_TESTS': 'JDK' + (unit.get('JDK_VERSION') or unit.get('JDK_REAL_VERSION') or '_DEFAULT') + '_FOR_TESTS',
        'YT-SPEC': serialize_list(get_unit_list_variable(unit, 'TEST_YT_SPEC_VALUE')),
    }
    test_classpath_origins = unit.get('TEST_CLASSPATH_VALUE')
    if test_classpath_origins:
        test_record['TEST_CLASSPATH_ORIGINS'] = test_classpath_origins
        test_record['TEST_CLASSPATH'] = '${TEST_CLASSPATH_MANAGED}'
    elif ymake_java_test:
        test_record['TEST_CLASSPATH'] = '${DART_CLASSPATH}'
        test_record['TEST_CLASSPATH_DEPS'] = '${DART_CLASSPATH_DEPS}'
        if unit.get('UNITTEST_DIR'):
            test_record['TEST_JAR'] = '${UNITTEST_MOD}'
        else:
            test_record['TEST_JAR'] = '{}/{}.jar'.format(unit.get('MODDIR'), unit.get('REALPRJNAME'))

    data = dump_test(unit, test_record)
    if data:
        unit.set_property(['DART_DATA', data])


def onjava_test_deps(unit, *args):
    if unit.get("TIDY") == "yes":
        # graph changed for clang_tidy tests
        return

    assert unit.get('MODULE_TYPE') is not None
    assert len(args) == 1
    mode = args[0]

    path = get_norm_unit_path(unit)
    ymake_java_test = unit.get('YMAKE_JAVA_TEST') == 'yes'

    test_record = {
        'SOURCE-FOLDER-PATH': path,
        'TEST-NAME': '-'.join([os.path.basename(os.path.dirname(path)), os.path.basename(path), 'dependencies']).strip('-'),
        'SCRIPT-REL-PATH': 'java.dependency.test',
        'TEST-TIMEOUT': '',
        'TESTED-PROJECT-NAME': path,
        'TEST-DATA': '',
        'TEST_PARTITION': 'SEQUENTIAL',
        'FORK-MODE': '',
        'SPLIT-FACTOR': '',
        'CUSTOM-DEPENDENCIES': ' '.join(get_values_list(unit, 'TEST_DEPENDS_VALUE')),
        'TAG': '',
        'SIZE': 'SMALL',
        'IGNORE_CLASSPATH_CLASH': ' '.join(get_values_list(unit, 'JAVA_IGNORE_CLASSPATH_CLASH_VALUE')),
        'NO_JBUILD': 'yes' if ymake_java_test else 'no',

        # JTEST/JTEST_FOR only
        'MODULE_TYPE': unit.get('MODULE_TYPE'),
        'UNITTEST_DIR': '',
        'SYSTEM_PROPERTIES': '',
        'TEST-CWD': '',
    }
    if mode == 'strict':
        test_record['STRICT_CLASSPATH_CLASH'] = 'yes'

    if ymake_java_test:
        test_record['CLASSPATH'] = '$B/{}/{}.jar ${{DART_CLASSPATH}}'.format(unit.get('MODDIR'), unit.get('REALPRJNAME'))

    data = dump_test(unit, test_record)
    unit.set_property(['DART_DATA', data])


def _get_test_tags(unit, spec_args=None):
    if spec_args is None:
        spec_args = {}
    tags = spec_args.get('TAG', []) + get_values_list(unit, 'TEST_TAGS_VALUE')
    # DEVTOOLS-7571
    if unit.get('SKIP_TEST_VALUE') and 'ya:fat' in tags and "ya:not_autocheck" not in tags:
        tags.append("ya:not_autocheck")

    return tags


def _dump_test(
        unit,
        test_type,
        test_files,
        timeout,
        test_dir,
        custom_deps,
        test_data,
        python_paths,
        split_factor,
        fork_mode,
        test_size,
        tags,
        requirements,
        binary_path='',
        old_pytest=False,
        test_cwd=None,
        runner_bin=None,
        yt_spec=None,
        data_files=None
):

    if test_type == "PY_TEST":
        script_rel_path = "py.test"
    else:
        script_rel_path = test_type

    unit_path = unit.path()
    fork_test_files = unit.get('FORK_TEST_FILES_MODE')
    fork_mode = ' '.join(fork_mode) if fork_mode else ''
    use_arcadia_python = unit.get('USE_ARCADIA_PYTHON')
    if test_cwd:
        test_cwd = test_cwd.replace("$TEST_CWD_VALUE", "").replace('"MACRO_CALLS_DELIM"', "").strip()
    test_name = os.path.basename(binary_path)
    test_record = {
        'TEST-NAME': os.path.splitext(test_name)[0],
        'TEST-TIMEOUT': timeout,
        'SCRIPT-REL-PATH': script_rel_path,
        'TESTED-PROJECT-NAME': test_name,
        'SOURCE-FOLDER-PATH': test_dir,
        'CUSTOM-DEPENDENCIES': " ".join(custom_deps),
        'TEST-ENV': prepare_env(unit.get("TEST_ENV_VALUE")),
        #  'TEST-PRESERVE-ENV': 'da',
        'TEST-DATA': serialize_list(sorted(_common.filter_out_by_keyword(test_data, 'AUTOUPDATED'))),
        'TEST-RECIPES': prepare_recipes(unit.get("TEST_RECIPES_VALUE")),
        'SPLIT-FACTOR': split_factor,
        'TEST_PARTITION': unit.get('TEST_PARTITION') or 'SEQUENTIAL',
        'FORK-MODE': fork_mode,
        'FORK-TEST-FILES': fork_test_files,
        'TEST-FILES': serialize_list(test_files),
        'SIZE': test_size,
        'TAG': serialize_list(tags),
        'REQUIREMENTS': serialize_list(requirements),
        'USE_ARCADIA_PYTHON': use_arcadia_python or '',
        'OLD_PYTEST': 'yes' if old_pytest else 'no',
        'PYTHON-PATHS': serialize_list(python_paths),
        'TEST-CWD': test_cwd or '',
        'SKIP_TEST': unit.get('SKIP_TEST_VALUE') or '',
        'BUILD-FOLDER-PATH': _common.strip_roots(unit_path),
        'BLOB': unit.get('TEST_BLOB_DATA') or '',
        'CANONIZE_SUB_PATH': unit.get('CANONIZE_SUB_PATH') or '',
    }
    if binary_path:
        test_record['BINARY-PATH'] = _common.strip_roots(binary_path)
    if runner_bin:
        test_record['TEST-RUNNER-BIN'] = runner_bin
    if yt_spec:
        test_record['YT-SPEC'] = serialize_list(yt_spec)
    data = dump_test(unit, test_record)
    if data:
        unit.set_property(["DART_DATA", data])
        save_in_file(unit.get('TEST_DART_OUT_FILE'), data)


def onsetup_pytest_bin(unit, *args):
    use_arcadia_python = unit.get('USE_ARCADIA_PYTHON') == "yes"
    if use_arcadia_python:
        unit.onresource(['-', 'PY_MAIN={}'.format("library.python.pytest.main:main")])  # XXX
        unit.onadd_pytest_bin(list(args))
    else:
        unit.onno_platform()
        unit.onadd_pytest_script(["PY_TEST"])


def onrun(unit, *args):
    exectest_cmd = unit.get(["EXECTEST_COMMAND_VALUE"]) or ''
    exectest_cmd += "\n" + subprocess.list2cmdline(args)
    unit.set(["EXECTEST_COMMAND_VALUE", exectest_cmd])


def onsetup_exectest(unit, *args):
    command = unit.get(["EXECTEST_COMMAND_VALUE"])
    if command is None:
        ymake.report_configure_error("EXECTEST must have at least one RUN macro")
        return
    command = command.replace("$EXECTEST_COMMAND_VALUE", "")
    if "PYTHON_BIN" in command:
        unit.ondepends('contrib/tools/python')
    unit.set(["TEST_BLOB_DATA", base64.b64encode(command)])
    add_test_to_dart(unit, "exectest", binary_path=os.path.join(unit.path(), unit.filename()).replace(".pkg", ""))


def onsetup_run_python(unit):
    if unit.get("USE_ARCADIA_PYTHON") == "yes":
        unit.ondepends('contrib/tools/python')


def get_canonical_test_resources(unit):
    unit_path = unit.path()
    canon_data_dir = os.path.join(unit.resolve(unit_path), CANON_DATA_DIR_NAME, unit.get('CANONIZE_SUB_PATH') or '')

    try:
        _, dirs, files = next(os.walk(canon_data_dir))
    except StopIteration:
        # path doesn't exist
        return [], []

    if CANON_RESULT_FILE_NAME in files:
        return _get_canonical_data_resources_v2(os.path.join(canon_data_dir, CANON_RESULT_FILE_NAME), unit_path)
    return [], []


def _load_canonical_file(filename, unit_path):
    try:
        with open(filename) as results_file:
            return json.load(results_file)
    except Exception as e:
        print>>sys.stderr, "malformed canonical data in {}: {} ({})".format(unit_path, e, filename)
        return {}


def _get_resource_from_uri(uri):
    m = CANON_MDS_RESOURCE_REGEX.match(uri)
    if m:
        res_id = m.group(1)
        return "{}:{}".format(MDS_SCHEME, res_id)

    m = CANON_SBR_RESOURCE_REGEX.match(uri)
    if m:
        # There might be conflict between resources, because all resources in sandbox have 'resource.tar.gz' name
        # That's why we use notation with '=' to specify specific path for resource
        uri = m.group(1)
        res_id = m.group(2)
        return "{}={}".format(uri, '/'.join([CANON_OUTPUT_STORAGE, res_id]))


def _get_external_resources_from_canon_data(data):
    # Method should work with both canonization versions:
    #   result.json: {'uri':X 'checksum':Y}
    #   result.json: {'testname': {'uri':X 'checksum':Y}}
    #   result.json: {'testname': [{'uri':X 'checksum':Y}]}
    # Also there is a bug - if user returns {'uri': 1} from test - machinery will fail
    # That's why we check 'uri' and 'checksum' fields presence
    # (it's still a bug - user can return {'uri':X, 'checksum': Y}, we need to unify canonization format)
    res = set()

    if isinstance(data, dict):
        if 'uri' in data and 'checksum' in data:
            resource = _get_resource_from_uri(data['uri'])
            if resource:
                res.add(resource)
        else:
            for k, v in data.iteritems():
                res.update(_get_external_resources_from_canon_data(v))
    elif isinstance(data, list):
        for e in data:
            res.update(_get_external_resources_from_canon_data(e))

    return res


def _get_canonical_data_resources_v2(filename, unit_path):
    return (_get_external_resources_from_canon_data(_load_canonical_file(filename, unit_path)), [filename])
