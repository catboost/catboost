import _common as common
import ymake
import json
import base64


DELIM = '================================'


def split_args(s):  # TODO quotes, escapes
    return filter(None, s.split())


def extract_macro_calls(unit, macro_value_name, macro_calls_delim):
    if not unit.get(macro_value_name):
        return []

    return filter(None, map(split_args, unit.get(macro_value_name).replace('$' + macro_value_name, '').split(macro_calls_delim)))


def extract_macro_calls2(unit, macro_value_name):
    if not unit.get(macro_value_name):
        return []

    calls = []
    for call_encoded_args in unit.get(macro_value_name).strip().split():
        call_args = json.loads(base64.b64decode(call_encoded_args), encoding='utf-8')
        calls.append(call_args)

    return calls


def onrun_java_program(unit, *args):
    """
    Custom code generation
    @link: https://wiki.yandex-team.ru/yatool/java/#kodogeneracijarunjavaprogram
    """
    flat, kv = common.sort_by_keywords(
        {'IN': -1, 'IN_DIR': -1, 'OUT': -1, 'OUT_DIR': -1, 'CWD': 1, 'CLASSPATH': -1, 'ADD_SRCS_TO_CLASSPATH': 0},
        args
    )

    for cp in kv.get('CLASSPATH', []):
        unit.oninternal_recurse(cp)

    prev = unit.get(['RUN_JAVA_PROGRAM_VALUE']) or ''
    new_val = (prev + ' ' + base64.b64encode(json.dumps(list(args), encoding='utf-8'))).strip()
    unit.set(['RUN_JAVA_PROGRAM_VALUE', new_val])


def ongenerate_script(unit, *args):
    """
    heretic@ promised to make tutorial here
    Don't forget
    Feel free to remind
    """
    flat, kv = common.sort_by_keywords(
        {'OUT': 1, 'TEMPLATE': -1, 'CUSTOM_PROPERTY': -1},
        args
    )
    if len(kv.get('TEMPLATE', [])) > 1:
        ymake.report_configure_error('To mane arguments for TEMPLATE parameter')
    prev = unit.get(['GENERATE_SCRIPT_VALUE']) or ''
    new_val = (prev + ' ' + base64.b64encode(json.dumps(list(args), encoding='utf-8'))).strip()
    unit.set(['GENERATE_SCRIPT_VALUE', new_val])


def onjava_module(unit, *args):
    unit.oninternal_recurse('contrib/java/org/sonarsource/scanner/cli/sonar-scanner-cli/2.8')  # TODO if <needs_sonar>

    if unit.get('COVERAGE'):
        unit.oninternal_recurse('devtools/jacoco-agent')

    args_delim = unit.get('ARGS_DELIM')

    data = {
        'PATH': unit.path(),
        'MODULE_TYPE': unit.get('MODULE_TYPE'),
        'MODULE_ARGS': unit.get('MODULE_ARGS'),
        'PEERDIR': unit.get_module_dirs('PEERDIRS'),
        'EXCLUDE': extract_macro_calls(unit, 'EXCLUDE_VALUE', args_delim),
        'JAVA_SRCS': extract_macro_calls(unit, 'JAVA_SRCS_VALUE', args_delim),
        'JAVAC_FLAGS': extract_macro_calls(unit, 'JAVAC_FLAGS_VALUE', args_delim),
        'ANNOTATION_PROCESSOR': extract_macro_calls(unit, 'ANNOTATION_PROCESSOR_VALUE', args_delim),
        'EXTERNAL_JAR': extract_macro_calls(unit, 'EXTERNAL_JAR_VALUE', args_delim),
        'RUN_JAVA_PROGRAM': extract_macro_calls2(unit, 'RUN_JAVA_PROGRAM_VALUE'),
        'ADD_WAR': extract_macro_calls(unit, 'ADD_WAR_VALUE', args_delim),
        'DEPENDENCY_MANAGEMENT': extract_macro_calls(unit, 'DEPENDENCY_MANAGEMENT_VALUE', args_delim),
        'MAVEN_GROUP_ID': extract_macro_calls(unit, 'MAVEN_GROUP_ID_VALUE', args_delim),

        # TODO remove when java test dart is in prod
        'UNITTEST_DIR': unit.get('UNITTEST_DIR'),
        'SYSTEM_PROPERTIES': extract_macro_calls(unit, 'SYSTEM_PROPERTIES_VALUE', args_delim),
        'JVM_ARGS': extract_macro_calls(unit, 'JVM_ARGS_VALUE', args_delim),
        'TEST_CWD': extract_macro_calls(unit, 'TEST_CWD_VALUE', args_delim),
        'TEST_DATA': extract_macro_calls(unit, '__test_data', args_delim),
        'TEST_FORK_MODE': extract_macro_calls(unit, 'TEST_FORK_MODE', args_delim),
        'SPLIT_FACTOR': extract_macro_calls(unit, 'TEST_SPLIT_FACTOR', args_delim),
        'TIMEOUT': extract_macro_calls(unit, 'TEST_TIMEOUT', args_delim),
        'TAG': extract_macro_calls(unit, 'TEST_TAGS_VALUE', args_delim),
        'SIZE': extract_macro_calls(unit, 'TEST_SIZE_NAME', args_delim),
        'DEPENDS': extract_macro_calls(unit, 'TEST_DEPENDS_VALUE', args_delim),
        'IDEA_EXCLUDE': extract_macro_calls(unit, 'IDEA_EXCLUDE_DIRS_VALUE', args_delim),
        'GENERATE_SCRIPT': extract_macro_calls2(unit, 'GENERATE_SCRIPT_VALUE'),
    }
    if unit.get('JAVA_ADD_DLLS_VALUE') == 'yes':
        data['ADD_DLLS_FROM_DEPENDS'] = extract_macro_calls(unit, 'JAVA_ADD_DLLS_VALUE', args_delim)

    if unit.get('ERROR_PRONE_VALUE') == 'yes':
        data['ERROR_PRONE'] = extract_macro_calls(unit, 'ERROR_PRONE_VALUE', args_delim)

    if unit.get('MAKE_UBERJAR_VALUE') == 'yes':
        if unit.get('MODULE_TYPE') != 'JAVA_PROGRAM':
            ymake.report_configure_error('{}: UBERJAR supported only for JAVA_PROGRAM module type'.format(unit.path()))
        data['UBERJAR'] = extract_macro_calls(unit, 'MAKE_UBERJAR_VALUE', args_delim)
        data['UBERJAR_PREFIX'] = extract_macro_calls(unit, 'UBERJAR_PREFIX_VALUE', args_delim)
        data['UBERJAR_HIDE_EXCLUDE'] = extract_macro_calls(unit, 'UBERJAR_HIDE_EXCLUDE_VALUE', args_delim)
        data['UBERJAR_PATH_EXCLUDE'] = extract_macro_calls(unit, 'UBERJAR_PATH_EXCLUDE_VALUE', args_delim)

    for dm_paths in data['DEPENDENCY_MANAGEMENT']:
        for p in dm_paths:
            unit.oninternal_recurse(p)

    for k, v in data.items():
        if not v:
            data.pop(k)

    dart = 'JAVA_DART: ' + base64.b64encode(json.dumps(data)) + '\n' + DELIM + '\n'

    unit.set_property(['JAVA_DART_DATA', dart])
    if unit.get('MODULE_TYPE') in ('JAVA_PROGRAM', 'JAVA_LIBRARY', 'JTEST', 'TESTNG') and not unit.path().startswith('$S/contrib/java'):
        if (unit.get('CHECK_JAVA_DEPS_VALUE') or '').lower() == 'yes':
            unit.onjava_test_deps()
        if unit.get('LINT_LEVEL_VALUE') != "none":
            unit.onadd_check(['JAVA_STYLE', unit.get('LINT_LEVEL_VALUE')])
