import _common as common


def onacceleo(unit, *args):
    flat, kv = common.sort_by_keywords(
        {'XSD': -1, 'MTL': -1, 'MTL_ROOT': 1, 'MTL_EXTENSION': -1, 'LANG': -1, 'OUT': -1, 'OUT_NOAUTO': -1, 'OUTPUT_INCLUDES': -1, 'DEBUG': 0},
        args
    )

    try:
        mtlroot = kv['MTL_ROOT'][0]
    except Exception:
        mtlroot = unit.path().replace('$S/', '')

    classpath = ['$SCARAB', ]  # XXX special word for ya make to replace following paths with real classpath
    classpath.append('tools/acceleo')
    classpath.extend(kv.get('MTL_EXTENSION', []))

    depends = []
    if not unit.get('IDE_MSVS_CALL'):
        for jar in classpath[1:]:
            depends.append(jar)

    classpath = ':'.join(classpath)

    # Generate java cmd
    cmd = [
        '-classpath',
        classpath,
        '-Dfile.encoding=UTF-8',
        'ru.yandex.se.logsng.tool.Cli',
    ]

    for xsd in kv.get('XSD', []):
        cmd += ['--xsd', xsd]

    for mtl in kv.get('MTL', []):
        cmd += ['--mtl', mtl]

    for lang in kv.get('LANG', []):
        cmd += ['--lang', lang]

    cmd += ['--output-dir', unit.path().replace('$S/', '${ARCADIA_BUILD_ROOT}/')]
    cmd += ['--build-root', '${ARCADIA_BUILD_ROOT}']
    cmd += ['--source-root', '${ARCADIA_ROOT}']
    cmd += ['--mtl-root', mtlroot]

    # Generate RUN_JAVA args
    run_java = cmd

    if 'DEBUG' not in kv:
        run_java += ['HIDE_OUTPUT']

    inputs = kv.get('XSD', []) + kv.get('MTL', []) + kv.get('LANG', [])
    if inputs:
        run_java += ['IN'] + inputs

    for k in 'OUT', 'OUT_NOAUTO', 'OUTPUT_INCLUDES':
        if kv.get(k):
            run_java += [k] + kv[k]

    if depends:
        run_java += ['TOOL'] + depends

    unit.on_run_java(run_java)
