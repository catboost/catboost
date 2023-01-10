import _common as common
import ymake
import json
import os
import base64


DELIM = '================================'
CONTRIB_JAVA_PREFIX = 'contrib/java/'


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


def on_run_jbuild_program(unit, *args):
    args = list(args)
    """
    Custom code generation
    @link: https://wiki.yandex-team.ru/yatool/java/#kodogeneracijarunjavaprogram
    """

    flat, kv = common.sort_by_keywords({'IN': -1, 'IN_DIR': -1, 'OUT': -1, 'OUT_DIR': -1, 'CWD': 1, 'CLASSPATH': -1, 'CP_USE_COMMAND_FILE': 1, 'ADD_SRCS_TO_CLASSPATH': 0}, args)
    depends = kv.get('CLASSPATH', []) + kv.get('JAR', [])
    fake_out = None
    if depends:
        # XXX: hack to force ymake to build dependencies
        fake_out = "fake.out.{}".format(hash(tuple(args)))
        unit.on_run_java(['TOOL'] + depends + ["OUT", fake_out])

    if not kv.get('CP_USE_COMMAND_FILE'):
        args += ['CP_USE_COMMAND_FILE', unit.get(['JAVA_PROGRAM_CP_USE_COMMAND_FILE']) or 'yes']

    if fake_out is not None:
        args += ['FAKE_OUT', fake_out]

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
    idea_only = True if 'IDEA_ONLY' in args else False

    if idea_only:
        if unit.get('YA_IDE_IDEA') != 'yes':
            return
        if unit.get('YMAKE_JAVA_MODULES') != 'yes':
            return

    data = {
        'BUNDLE_NAME': unit.name(),
        'PATH': unit.path(),
        'IDEA_ONLY': 'yes' if idea_only else 'no',
        'MODULE_TYPE': unit.get('MODULE_TYPE'),
        'MODULE_ARGS': unit.get('MODULE_ARGS'),
        'MANAGED_PEERS': '${MANAGED_PEERS}',
        'MANAGED_PEERS_CLOSURE': '${MANAGED_PEERS_CLOSURE}',
        'NON_NAMAGEABLE_PEERS': '${NON_NAMAGEABLE_PEERS}',
        'TEST_CLASSPATH_MANAGED': '${TEST_CLASSPATH_MANAGED}',
        'EXCLUDE': extract_macro_calls(unit, 'EXCLUDE_VALUE', args_delim),
        'JAVA_SRCS': extract_macro_calls(unit, 'JAVA_SRCS_VALUE', args_delim),
        'JAVAC_FLAGS': extract_macro_calls(unit, 'JAVAC_FLAGS_VALUE', args_delim),
        'ANNOTATION_PROCESSOR': extract_macro_calls(unit, 'ANNOTATION_PROCESSOR_VALUE', args_delim),
        'EXTERNAL_JAR': extract_macro_calls(unit, 'EXTERNAL_JAR_VALUE', args_delim),
        'RUN_JAVA_PROGRAM': extract_macro_calls2(unit, 'RUN_JAVA_PROGRAM_VALUE'),
        'RUN_JAVA_PROGRAM_MANAGED': '${RUN_JAVA_PROGRAM_MANAGED}',
        'MAVEN_GROUP_ID': extract_macro_calls(unit, 'MAVEN_GROUP_ID_VALUE', args_delim),
        'JAR_INCLUDE_FILTER': extract_macro_calls(unit, 'JAR_INCLUDE_FILTER_VALUE', args_delim),
        'JAR_EXCLUDE_FILTER': extract_macro_calls(unit, 'JAR_EXCLUDE_FILTER_VALUE', args_delim),

        # TODO remove when java test dart is in prod
        'UNITTEST_DIR': unit.get('UNITTEST_DIR'),
        'SYSTEM_PROPERTIES': extract_macro_calls(unit, 'SYSTEM_PROPERTIES_VALUE', args_delim),
        'JVM_ARGS': extract_macro_calls(unit, 'JVM_ARGS_VALUE', args_delim),
        'TEST_CWD': extract_macro_calls(unit, 'TEST_CWD_VALUE', args_delim),
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
        'JDK_RESOURCE': 'JDK' + (unit.get('JDK_VERSION')  or unit.get('JDK_REAL_VERSION') or '_DEFAULT')
    }
    if unit.get('ENABLE_PREVIEW_VALUE') == 'yes' and (unit.get('JDK_VERSION') or unit.get('JDK_REAL_VERSION')) in ('15', '16', '17', '18', '19'):
        data['ENABLE_PREVIEW'] = extract_macro_calls(unit, 'ENABLE_PREVIEW_VALUE', args_delim)

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
        if unit.get('KOTLINC_FLAGS_VALUE'):
            data['KOTLINC_FLAGS'] = extract_macro_calls(unit, 'KOTLINC_FLAGS_VALUE', args_delim)
        if unit.get('KOTLINC_OPTS_VALUE'):
            data['KOTLINC_OPTS'] = extract_macro_calls(unit, 'KOTLINC_OPTS_VALUE', args_delim)

    if unit.get('DIRECT_DEPS_ONLY_VALUE') == 'yes':
        data['DIRECT_DEPS_ONLY'] = extract_macro_calls(unit, 'DIRECT_DEPS_ONLY_VALUE', args_delim)

    if unit.get('JAVA_EXTERNAL_DEPENDENCIES_VALUE'):
        valid = []
        for dep in sum(extract_macro_calls(unit, 'JAVA_EXTERNAL_DEPENDENCIES_VALUE', args_delim), []):
            if os.path.normpath(dep).startswith('..'):
                ymake.report_configure_error('{}: {} - relative paths in JAVA_EXTERNAL_DEPENDENCIES is not allowed'.format(unit.path(), dep))
            elif os.path.isabs(dep):
                ymake.report_configure_error('{}: {} absolute paths in JAVA_EXTERNAL_DEPENDENCIES is not allowed'.format(unit.path(), dep))
            else:
                valid.append(dep)
        if valid:
            data['EXTERNAL_DEPENDENCIES'] = [valid]

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

    for k, v in data.items():
        if not v:
            data.pop(k)

    dart = 'JAVA_DART: ' + base64.b64encode(json.dumps(data)) + '\n' + DELIM + '\n'

    unit.set_property(['JAVA_DART_DATA', dart])
    if not idea_only and unit.get('MODULE_TYPE') in ('JAVA_PROGRAM', 'JAVA_LIBRARY', 'JTEST', 'TESTNG', 'JUNIT5') and not unit.path().startswith('$S/contrib/java'):
        unit.on_add_classpath_clash_check()
        if unit.get('LINT_LEVEL_VALUE') != "none" and unit.get('_NO_LINT_VALUE') != 'none':
            unit.onadd_check(['JAVA_STYLE', unit.get('LINT_LEVEL_VALUE')])


def on_add_java_style_checks(unit, *args):
    if unit.get('LINT_LEVEL_VALUE') != "none" and unit.get('_NO_LINT_VALUE') != 'none':
        unit.onadd_check(['JAVA_STYLE', unit.get('LINT_LEVEL_VALUE')] + list(args))


def on_add_kotlin_style_checks(unit, *args):
    """
    ktlint can be disabled using NO_LINT() and NO_LINT(ktlint)
    """
    if unit.get('WITH_KOTLIN_VALUE') == 'yes':
        no_lint_value = unit.get('_NO_LINT_VALUE')
        if no_lint_value == '':
            unit.onadd_check(['ktlint'] + list(args))
        elif no_lint_value not in ('none', 'none_internal', 'ktlint'):
            ymake.report_configure_error('Unsupported value for NO_LINT macro: {}'.format(no_lint_value))



def on_add_classpath_clash_check(unit, *args):
    jdeps_val = (unit.get('CHECK_JAVA_DEPS_VALUE') or '').lower()
    if jdeps_val and jdeps_val not in ('yes', 'no', 'strict'):
        ymake.report_configure_error('CHECK_JAVA_DEPS: "yes", "no" or "strict" required')
    if jdeps_val and jdeps_val != 'no':
        unit.onjava_test_deps(jdeps_val)


# Ymake java modules related macroses


def onexternal_jar(unit, *args):
    args = list(args)
    flat, kv = common.sort_by_keywords({'SOURCES': 1}, args)
    if not flat:
        ymake.report_configure_error('EXTERNAL_JAR requires exactly one resource URL of compiled jar library')
    res = flat[0]
    resid = res[4:] if res.startswith('sbr:') else res
    unit.set(['JAR_LIB_RESOURCE', resid])
    unit.set(['JAR_LIB_RESOURCE_URL', res])


def on_check_java_srcdir(unit, *args):
    args = list(args)
    for arg in args:
        if not '$' in arg:
            arc_srcdir = os.path.join(unit.get('MODDIR'), arg)
            abs_srcdir = unit.resolve(os.path.join("$S/", arc_srcdir))
            if not os.path.exists(abs_srcdir) or not os.path.isdir(abs_srcdir):
                ymake.report_configure_error(
                    'Trying to set a [[alt1]]JAVA_SRCS[[rst]] for a missing directory: [[imp]]$S/{}[[rst]]',
                    missing_dir=arc_srcdir
                )
            return
        srcdir = unit.resolve_arc_path(arg)
        if srcdir and not srcdir.startswith('$S'):
            continue
        abs_srcdir = unit.resolve(srcdir) if srcdir else unit.resolve(arg)
        if not os.path.exists(abs_srcdir) or not os.path.isdir(abs_srcdir):
            ymake.report_configure_error(
                'Trying to set a [[alt1]]JAVA_SRCS[[rst]] for a missing directory: [[imp]]{}[[rst]]',
                missing_dir=srcdir
            )


def on_fill_jar_copy_resources_cmd(unit, *args):
    if len(args) == 4:
        varname, srcdir, base_classes_dir, reslist = tuple(args)
        package = ''
    else:
        varname, srcdir, base_classes_dir, package, reslist = tuple(args)
    dest_dir = os.path.join(base_classes_dir, *package.split('.')) if package else base_classes_dir
    var = unit.get(varname)
    var += ' && $FS_TOOLS copy_files {} {} {}'.format(srcdir if srcdir.startswith('"$') else '${CURDIR}/' + srcdir, dest_dir, reslist)
    unit.set([varname, var])


def on_fill_jar_gen_srcs(unit, *args):
    varname, jar_type, srcdir, base_classes_dir, java_list, kt_list, groovy_list, res_list = tuple(args[0:8])
    resolved_srcdir = unit.resolve_arc_path(srcdir)
    if not resolved_srcdir.startswith('$') or resolved_srcdir.startswith('$S'):
        return

    exclude_pos = args.index('EXCLUDE')
    globs = args[7:exclude_pos]
    excludes = args[exclude_pos + 1:]
    var = unit.get(varname)
    var += ' && ${{cwd:BINDIR}} $YMAKE_PYTHON ${{input:"build/scripts/resolve_java_srcs.py"}} --append -d {} -s {} -k {} -g {} -r {} --include-patterns {}'.format(srcdir, java_list, kt_list, groovy_list, res_list, ' '.join(globs))
    if jar_type == 'SRC_JAR':
        var += ' --all-resources'
    if len(excludes) > 0:
        var += ' --exclude-patterns {}'.format(' '.join(excludes))
    if unit.get('WITH_KOTLIN_VALUE') == 'yes':
        var += ' --resolve-kotlin'
    unit.set([varname, var])


def on_check_run_java_prog_classpath(unit, *args):
    if len(args) != 1:
        ymake.report_configure_error('multiple CLASSPATH elements in RUN_JAVA_PROGRAM invocation no more supported. Use JAVA_RUNTIME_PEERDIR on the JAVA_PROGRAM module instead')


def extract_words(words, keys):
    kv = {}
    k = None

    for w in words:
        if w in keys:
            k = w
        else:
            if not k in kv:
                kv[k] = []
            kv[k].append(w)

    return kv


def parse_words(words):
    kv = extract_words(words, {'OUT', 'TEMPLATE'})
    if not 'TEMPLATE' in kv:
        kv['TEMPLATE'] = ['template.tmpl']
    ws = []
    for item in ('OUT', 'TEMPLATE'):
        for i, word in list(enumerate(kv[item])):
            if word == 'CUSTOM_PROPERTY':
                ws += kv[item][i:]
                kv[item] = kv[item][:i]
    templates = kv['TEMPLATE']
    outputs = kv['OUT']
    if len(outputs) < len(templates):
        ymake.report_configure_error('To many arguments for TEMPLATE parameter')
        return
    if ws and ws[0] != 'CUSTOM_PROPERTY':
        ymake.report_configure_error('''Can't parse {}'''.format(ws))
    custom_props = []
    for item in ws:
        if item == 'CUSTOM_PROPERTY':
            custom_props.append([])
        else:
            custom_props[-1].append(item)
    props = []
    for p in custom_props:
        if not p:
            ymake.report_configure_error('Empty CUSTOM_PROPERTY')
            continue
        props.append('-B')
        if len(p) > 1:
            props.append(base64.b64encode("{}={}".format(p[0], ' '.join(p[1:]))))
        else:
            ymake.report_configure_error('CUSTOM_PROPERTY "{}" value is not specified'.format(p[0]))
    for i, o in enumerate(outputs):
        yield o, templates[min(i, len(templates) - 1)], props


def on_ymake_generate_script(unit, *args):
    for out, tmpl, props in parse_words(list(args)):
        unit.on_add_gen_java_script([out, tmpl] + list(props))


def on_jdk_version_macro_check(unit, *args):
    if len(args) != 1:
        unit.message(["error", "Invalid syntax. Single argument required."])
    jdk_version = args[0]
    available_versions = ('10', '11', '15', '16', '17', '18', '19',)
    if jdk_version not in available_versions:
        ymake.report_configure_error("Invalid jdk version: {}. {} are available".format(jdk_version, available_versions))
    if int(jdk_version) >= 19 and unit.get('WITH_JDK_VALUE') != 'yes' and unit.get('MODULE_TAG') == 'JAR_RUNNABLE':
        msg = (
            "Missing WITH_JDK() macro for JDK version >= 19"
            # temporary link with additional explanation
            ". For more info see https://clubs.at.yandex-team.ru/arcadia/28543"
        )
        ymake.report_configure_error(msg)


def _maven_coords_for_project(unit, project_dir):
    parts = project_dir.split('/')

    g = '.'.join(parts[2:-2])
    a = parts[-2]
    v = parts[-1]
    c = ''

    pom_path = unit.resolve(os.path.join('$S', project_dir, 'pom.xml'))
    if os.path.exists(pom_path):
        import xml.etree.ElementTree as et
        with open(pom_path) as f:
            root = et.fromstring(f.read())
        for xpath in ('./{http://maven.apache.org/POM/4.0.0}artifactId', './artifactId'):
            artifact = root.find(xpath)
            if artifact is not None:
                artifact = artifact.text
                if a != artifact and a.startswith(artifact):
                    c = a[len(artifact):].lstrip('-_')
                    a = artifact
                break

    return '{}:{}:{}:{}'.format(g, a, v, c)


def on_setup_maven_export_coords_if_need(unit, *args):
    if not unit.enabled('MAVEN_EXPORT'):
        return

    unit.set(['MAVEN_EXPORT_COORDS_GLOBAL', _maven_coords_for_project(unit, args[0])])


def on_setup_project_coords_if_needed(unit, *args):
    if not unit.enabled('EXPORT_GRADLE'):
        return

    project_dir = args[0]
    if project_dir.startswith(CONTRIB_JAVA_PREFIX):
        value = '\\"{}\\"'.format(_maven_coords_for_project(unit, project_dir).rstrip(':'))
    else:
        value = 'project(\\":{}\\")'.format(project_dir.replace('/', ':'))
    unit.set(['_EXPORT_GRADLE_PROJECT_COORDS', value])
