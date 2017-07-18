from _common import sort_by_keywords


def get_or_default(kv, name, default):
    if name in kv:
        return kv[name][0]
    return default


def onyabs_generate_conf(unit, *args):
    flat, kv = sort_by_keywords(
        {'MODE': 1, 'SCRIPT': 1, 'SRC': 1, 'TOOL': 1, 'CONF_DIR': 1, 'DEST': 1}, args
    )
    src = get_or_default(kv, 'SRC', 'yabs/server/phantom')
    mode = get_or_default(kv, 'MODE', 'production')

    script = src + "/" + get_or_default(kv, 'SCRIPT', 'mkconf.py')
    conf = src + "/" + get_or_default(kv, 'CONF_DIR', 'conf-tmpl')
    tool = src + "/" + get_or_default(kv, 'TOOL', 'yabs_conf')

    for name in flat:
        filename = "/".join([conf, name])
        unit.onbuiltin_python([
            script,
            "--cluster-conf-binary", tool,
            "--mode", mode,
            "--dest-dir", "${BINDIR}",
            filename,
            "IN", filename,
            "OUT", "${BINDIR}/%s" % name,
            "TOOL", tool
        ])


def onyabs_generate_phantom_conf_patch(unit, *args):
    flat, kv = sort_by_keywords(
        {'SRC': 1, 'DST': 1}, args
    )
    src = '${ARCADIA_BUILD_ROOT}/' + get_or_default(kv, 'SRC', 'yabs/server/phantom/conf')
    dst = '${ARCADIA_BUILD_ROOT}/' + get_or_default(kv, 'DST', 'yabs/server/phantom/conf-test')
    for f in flat:
        lhs = src + '/' + f
        rhs = dst + '/' + f
        unit.onbuiltin_python([
            'mkdiff.py',
            lhs, rhs,
            'IN', lhs,
            'IN', rhs,
            'STDOUT', f + ".patch"
        ])


def onyabs_generate_phantom_conf_test_check(unit, *args):
    yabs_path = args[0]
    for name in args[1:]:
        unit.onbuiltin_python("""
    build/scripts/wrapper.py mkcheckconf.sh ${{ARCADIA_BUILD_ROOT}}/{yabs_path}/phantom/conf-test/yabs-{role}.conf yabs-check-{role}.conf
    IN mkcheckconf.sh ${{ARCADIA_BUILD_ROOT}}/{yabs_path}/phantom/conf-test/yabs-{role}.conf
    OUT yabs-check-{role}.conf
""".format(yabs_path=yabs_path, role=name).split()  # noqa
        )
