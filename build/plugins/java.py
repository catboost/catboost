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
    args = list(args)
    """
    Custom code generation
    @link: https://wiki.yandex-team.ru/yatool/java/#kodogeneracijarunjavaprogram
    """

    flat, kv = common.sort_by_keywords({'IN': -1, 'IN_DIR': -1, 'OUT': -1, 'OUT_DIR': -1, 'CWD': 1, 'CLASSPATH': -1, 'CP_USE_COMMAND_FILE': 1, 'ADD_SRCS_TO_CLASSPATH': 0}, args)
    depends = kv.get('CLASSPATH', []) + kv.get('JAR', [])
    if depends:
        # XXX: hack to force ymake to build dependencies
        unit.on_run_java(['TOOL'] + depends + ["OUT", "fake.out.{}".format(hash(tuple(args)))])

    if not kv.get('CP_USE_COMMAND_FILE'):
       args += ['CP_USE_COMMAND_FILE', unit.get(['JAVA_PROGRAM_CP_USE_COMMAND_FILE']) or 'yes']

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
        {'OUT': -1, 'TEMPLATE': -1, 'CUSTOM_PROPERTY': -1},
        args
    )
    if len(kv.get('TEMPLATE', [])) > len(kv.get('OUT', [])):
        ymake.report_configure_error('To many arguments for TEMPLATE parameter')
    prev = unit.get(['GENERATE_SCRIPT_VALUE']) or ''
    new_val = (prev + ' ' + base64.b64encode(json.dumps(list(args), encoding='utf-8'))).strip()
    unit.set(['GENERATE_SCRIPT_VALUE', new_val])


def onjava_module(unit, *args):
    args_delim = unit.get('ARGS_DELIM')

    data = {
        'BUNDLE_NAME': unit.name(),
        'PATH': unit.path(),
        'MODULE_TYPE': unit.get('MODULE_TYPE'),
        'MODULE_ARGS': unit.get('MODULE_ARGS'),
        'PEERDIR': unit.get_module_dirs('PEERDIRS'),
        'MANAGED_PEERS': '${MANAGED_PEERS}',
        'MANAGED_PEERS_CLOSURE': '${MANAGED_PEERS_CLOSURE}',
        'EXCLUDE': extract_macro_calls(unit, 'EXCLUDE_VALUE', args_delim),
        'JAVA_SRCS': extract_macro_calls(unit, 'JAVA_SRCS_VALUE', args_delim),
        'JAVAC_FLAGS': extract_macro_calls(unit, 'JAVAC_FLAGS_VALUE', args_delim),
        'ANNOTATION_PROCESSOR': extract_macro_calls(unit, 'ANNOTATION_PROCESSOR_VALUE', args_delim),
        'EXTERNAL_JAR': extract_macro_calls(unit, 'EXTERNAL_JAR_VALUE', args_delim),
        'RUN_JAVA_PROGRAM': extract_macro_calls2(unit, 'RUN_JAVA_PROGRAM_VALUE'),
        'ADD_WAR': extract_macro_calls(unit, 'ADD_WAR_VALUE', args_delim),
        'DEPENDENCY_MANAGEMENT': extract_macro_calls(unit, 'DEPENDENCY_MANAGEMENT_VALUE', args_delim),
        'MAVEN_GROUP_ID': extract_macro_calls(unit, 'MAVEN_GROUP_ID_VALUE', args_delim),
        'JAR_INCLUDE_FILTER': extract_macro_calls(unit, 'JAR_INCLUDE_FILTER_VALUE', args_delim),
        'JAR_EXCLUDE_FILTER': extract_macro_calls(unit, 'JAR_EXCLUDE_FILTER_VALUE', args_delim),

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
        'IDEA_RESOURCE': extract_macro_calls(unit, 'IDEA_RESOURCE_DIRS_VALUE', args_delim),
        'IDEA_MODULE_NAME': extract_macro_calls(unit, 'IDEA_MODULE_NAME_VALUE', args_delim),
        'GENERATE_SCRIPT': extract_macro_calls2(unit, 'GENERATE_SCRIPT_VALUE'),
        'FAKEID': extract_macro_calls(unit, 'FAKEID', args_delim),
        'TEST_DATA': extract_macro_calls(unit, 'TEST_DATA_VALUE', args_delim),
        'JAVA_FORBIDDEN_LIBRARIES': extract_macro_calls(unit, 'JAVA_FORBIDDEN_LIBRARIES_VALUE', args_delim),
    }
    if unit.get('SAVE_JAVAC_GENERATED_SRCS_DIR') and unit.get('SAVE_JAVAC_GENERATED_SRCS_TAR'):
        data['SAVE_JAVAC_GENERATED_SRCS_DIR'] = extract_macro_calls(unit, 'SAVE_JAVAC_GENERATED_SRCS_DIR', args_delim)
        data['SAVE_JAVAC_GENERATED_SRCS_TAR'] = extract_macro_calls(unit, 'SAVE_JAVAC_GENERATED_SRCS_TAR', args_delim)

    if unit.get('JAVA_ADD_DLLS_VALUE') == 'yes':
        data['ADD_DLLS_FROM_DEPENDS'] = extract_macro_calls(unit, 'JAVA_ADD_DLLS_VALUE', args_delim)

    if unit.get('ERROR_PRONE_VALUE') == 'yes':
        data['ERROR_PRONE'] = extract_macro_calls(unit, 'ERROR_PRONE_VALUE', args_delim)

    if unit.get('WITH_KOTLIN_VALUE') == 'yes':
        data['WITH_KOTLIN'] = extract_macro_calls(unit, 'WITH_KOTLIN_VALUE', args_delim)
        if unit.get('KOTLIN_JVM_TARGET'):
            data['KOTLIN_JVM_TARGET'] = extract_macro_calls(unit, 'KOTLIN_JVM_TARGET', args_delim)

    if unit.get('DIRECT_DEPS_ONLY_VALUE') == 'yes':
        data['DIRECT_DEPS_ONLY'] = extract_macro_calls(unit, 'DIRECT_DEPS_ONLY_VALUE', args_delim)

    if unit.get('MAKE_UBERJAR_VALUE') == 'yes':
        if unit.get('MODULE_TYPE') != 'JAVA_PROGRAM':
            ymake.report_configure_error('{}: UBERJAR supported only for JAVA_PROGRAM module type'.format(unit.path()))
        data['UBERJAR'] = extract_macro_calls(unit, 'MAKE_UBERJAR_VALUE', args_delim)
        data['UBERJAR_PREFIX'] = extract_macro_calls(unit, 'UBERJAR_PREFIX_VALUE', args_delim)
        data['UBERJAR_HIDE_EXCLUDE'] = extract_macro_calls(unit, 'UBERJAR_HIDE_EXCLUDE_VALUE', args_delim)
        data['UBERJAR_PATH_EXCLUDE'] = extract_macro_calls(unit, 'UBERJAR_PATH_EXCLUDE_VALUE', args_delim)
        data['UBERJAR_MANIFEST_TRANSFORMER_MAIN'] = extract_macro_calls(unit, 'UBERJAR_MANIFEST_TRANSFORMER_MAIN_VALUE', args_delim)
        data['UBERJAR_MANIFEST_TRANSFORMER_ATTRIBUTE'] = extract_macro_calls(unit, 'UBERJAR_MANIFEST_TRANSFORMER_ATTRIBUTE_VALUE', args_delim)
        data['UBERJAR_APPENDING_TRANSFORMER'] = extract_macro_calls(unit, 'UBERJAR_APPENDING_TRANSFORMER_VALUE', args_delim)
        data['UBERJAR_SERVICES_RESOURCE_TRANSFORMER'] = extract_macro_calls(unit, 'UBERJAR_SERVICES_RESOURCE_TRANSFORMER_VALUE', args_delim)

    if unit.get('WITH_JDK_VALUE') == 'yes':
        if unit.get('MODULE_TYPE') != 'JAVA_PROGRAM':
            ymake.report_configure_error('{}: JDK export supported only for JAVA_PROGRAM module type'.format(unit.path()))
        data['WITH_JDK'] = extract_macro_calls(unit, 'WITH_JDK_VALUE', args_delim)

    for dm_paths in data['DEPENDENCY_MANAGEMENT']:
        for p in dm_paths:
            unit.on_ghost_peerdir(p)

    if not data['EXTERNAL_JAR']:
        has_processor = extract_macro_calls(unit, 'GENERATE_VCS_JAVA_INFO_NODEP', args_delim)
        data['EMBED_VCS'] = [[str(has_processor and has_processor[0] and has_processor[0][0])]]
        # FORCE_VCS_INFO_UPDATE is responsible for setting special value of VCS_INFO_DISABLE_CACHE__NO_UID__
        macro_val = extract_macro_calls(unit, 'FORCE_VCS_INFO_UPDATE', args_delim)
        macro_str = macro_val[0][0] if macro_val and macro_val[0] and macro_val[0][0] else ''
        if macro_str and macro_str == 'yes':
            data['VCS_INFO_DISABLE_CACHE__NO_UID__'] = macro_val

    for java_srcs_args in data['JAVA_SRCS']:
        external = None

        for i in xrange(len(java_srcs_args)):
            arg = java_srcs_args[i]

            if arg == 'EXTERNAL':
                if not i + 1 < len(java_srcs_args):
                    continue  # TODO configure error

                ex = java_srcs_args[i + 1]

                if ex in ('EXTERNAL', 'SRCDIR', 'PACKAGE_PREFIX', 'EXCLUDE'):
                    continue  # TODO configure error

                if external is not None:
                    continue  # TODO configure error

                external = ex

        if external:
            unit.onpeerdir(external)

    dep_veto = extract_macro_calls(unit, 'JAVA_DEPENDENCIES_CONFIGURATION_VALUE', args_delim)
    if dep_veto:
        dep_veto = set(dep_veto[0])
        if (unit.get('IGNORE_JAVA_DEPENDENCIES_CONFIGURATION') or '').lower() != 'yes':
            for veto in map(str.upper, dep_veto):
                if veto.upper() == 'FORBID_DIRECT_PEERDIRS':
                    data['JAVA_DEPENDENCY_DIRECT'] = [['yes']]
                elif veto.upper() == 'FORBID_DEFAULT_VERSIONS':
                    data['JAVA_DEPENDENCY_DEFAULT_VERSION'] = [['yes']]
                elif veto.upper() == 'FORBID_CONFLICT':
                    data['JAVA_DEPENDENCY_CHECK_RESOLVED_CONFLICTS'] = [['yes']]
                elif veto.upper() == 'FORBID_CONFLICT_DM':
                    data['JAVA_DEPENDENCY_DM_CHECK_DIFFERENT'] = [['yes']]
                elif veto.upper() == 'FORBID_CONFLICT_DM_RECENT':
                    data['JAVA_DEPENDENCY_DM_CHECK_RECENT'] = [['yes']]
                elif veto.upper() == 'REQUIRE_DM':
                    data['JAVA_DEPENDENCY_DM_REQUIRED'] = [['yes']]
                else:
                    ymake.report_configure_error('Unknown JAVA_DEPENDENCIES_CONFIGURATION value {} Allowed only [{}]'.format(veto, ', '.join(
                        ['FORBID_DIRECT_PEERDIRS', 'FORBID_DEFAULT_VERSIONS', 'FORBID_CONFLICT', 'FORBID_CONFLICT_DM', 'FORBID_CONFLICT_DM_RECENT', 'REQUIRE_DM']
                    )))

    for k, v in data.items():
        if not v:
            data.pop(k)

    dart = 'JAVA_DART: ' + base64.b64encode(json.dumps(data)) + '\n' + DELIM + '\n'

    unit.set_property(['JAVA_DART_DATA', dart])
    if unit.get('MODULE_TYPE') in ('JAVA_PROGRAM', 'JAVA_LIBRARY', 'JTEST', 'TESTNG', 'JUNIT5') and not unit.path().startswith('$S/contrib/java'):
        jdeps_val = (unit.get('CHECK_JAVA_DEPS_VALUE') or '').lower()
        if jdeps_val and jdeps_val not in ('yes', 'no', 'strict'):
            ymake.report_configure_error('CHECK_JAVA_DEPS: "yes", "no" or "strict" required')
        if jdeps_val and jdeps_val != 'no':
            unit.onjava_test_deps(jdeps_val)
        if unit.get('LINT_LEVEL_VALUE') != "none":
            unit.onadd_check(['JAVA_STYLE', unit.get('LINT_LEVEL_VALUE')])
