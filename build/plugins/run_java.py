import _common as common
import ymake


def onrun_java(unit, *args):
    flat, kv = common.sort_by_keywords(
        {'CLASSPATH': -1, 'IN': -1, 'OUT': -1, 'OUT_NOAUTO': -1, 'OUTPUT_INCLUDES': -1, 'DEBUG': 0, 'JAR': 1},
        args
    )
    if not (kv.get('CLASSPATH', []) + kv.get('JAR', [])):
        ymake.report_configure_error('Java program for RUN_JAVA is not specified')

    depends = []
    if not unit.get('IDE_MSVS_CALL'):
        for jar in (kv.get('CLASSPATH', []) + kv.get('JAR', [])):
            depends.append(jar)

    classpath = ':'.join(classpath)

    # Generate java cmd
    cmd = []
    if kv.get('JAR'):
        cmd += [
            '-jar',
            ':'.join(['$SCARAB_SLIM'] + kv.get('JAR')),
        ]
    cmd += [
        '-classpath',
        ':'.join(['$SCARAB'] + kv.get('JAR', []) + kv.get('CLASSPATH', [])),
        '-Dfile.encoding=UTF-8',
    ]

    cmd += flat

    if 'DEBUG' not in kv:
        cmd += ['HIDE_OUTPUT']

    for k in 'IN', 'OUT', 'OUT_NOAUTO', 'OUTPUT_INCLUDES':
        if kv.get(k):
            cmd += [k] + kv[k]

    if depends:
        cmd += ['TOOL'] + depends

    unit.on_run_java(cmd)
