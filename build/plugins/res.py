from _common import iterpair, listid, pathid, rootrel_arc_src, tobuilddir, filter_out_by_keyword


def split(lst, limit):
    # paths are specified with replaceable prefix
    # real length is unknown at the moment, that why we use root_lenght
    # as a rough estimation
    root_lenght = 200
    filepath = None
    lenght = 0
    bucket = []

    for item in lst:
        if filepath:
            lenght += root_lenght + len(filepath) + len(item)
            if lenght > limit and bucket:
                yield bucket
                bucket = []
                lenght = 0

            bucket.append(filepath)
            bucket.append(item)
            filepath = None
        else:
            filepath = item

    if bucket:
        yield bucket


def onfat_resource(unit, *args):
    unit.onpeerdir(['library/resource'])

    # Since the maximum length of lpCommandLine string for CreateProcess is 8kb (windows) characters,
    # we make several calls of rescompiler
    # https://msdn.microsoft.com/ru-ru/library/windows/desktop/ms682425.aspx
    for part_args in split(args, 8000):
        output = listid(part_args) + '.cpp'
        inputs = [x for x, y in iterpair(part_args) if x != '-']
        if inputs:
            inputs = ['IN'] + inputs

        unit.onrun_program(['tools/rescompiler', output] + part_args + inputs + ['OUT_NOAUTO', output])
        unit.onsrcs(['GLOBAL', output])


def gen_ro_flags(unit):
    if unit.enabled('darwin') or (unit.enabled('windows') and unit.enabled('arch_type_32')):
        return ['--prefix']
    return []


def onresource(unit, *args):
    unit.onpeerdir(['library/resource'])

    outs = []

    for part_args in split(args, 8000):
        srcs_gen = []
        raw_gen = []
        raw_inputs = []
        compressed = []
        compressed_input = []
        compressed_output = []
        for p, n in iterpair(part_args):
            if unit.enabled('ARCH_AARCH64') or unit.enabled('ARCH_ARM') or unit.enabled('ARCH_PPC64LE'):
                raw_gen += [p, n]
                if p != '-':
                    raw_inputs.append(p)
                continue
            lid = '_' + pathid(p + n + unit.path())
            output = lid + '.rodata'
            if p == '-':
                n, p = n.split('=', 1)
                compressed += ['-', p, output]
            else:
                compressed += [p, output]
                compressed_input.append(p)
                compressed_output.append(output)
            srcs_gen.append('{}={}'.format(lid, n))

        if compressed:
            lid = listid(part_args)
            fake_yasm = '_' + lid + '.yasm'
            cmd = ['tools/rescompressor', fake_yasm] + gen_ro_flags(unit) + compressed
            if compressed_input:
                cmd += ['IN'] + compressed_input
            cmd += ['OUT_NOAUTO', fake_yasm] + compressed_output
            unit.onrun_program(cmd)
            unit.onsrcs(['GLOBAL', tobuilddir(unit.path() + '/' + fake_yasm)])

        if srcs_gen:
            output = listid(part_args) + '.cpp'
            unit.onrun_program(['tools/rorescompiler', output] + srcs_gen + ['OUT_NOAUTO', output])
            outs.append(output)

        if raw_gen:
            output = listid(part_args) + '_raw.cpp'
            if raw_inputs:
                raw_inputs = ['IN'] + raw_inputs
            unit.onrun_program(['tools/rescompiler', output] + raw_gen + raw_inputs + ['OUT_NOAUTO', output])
            unit.onsrcs(['GLOBAL', output])

    if outs:
        if len(outs) > 1:
            unit.onjoin_srcs_global(['join_' + listid(outs) + '.cpp'] + outs)
        else:
            unit.onsrcs(['GLOBAL'] + outs)


def onfrom_sandbox(unit, *args):
    unit.onsetup_from_sandbox(filter_out_by_keyword(list(args), 'AUTOUPDATED'))
    res_id = args[0]
    if res_id == "FILE":
        res_id = args[1]
    unit.onadd_check(["check.resource", res_id])


def onresource_files(unit, *args):
    """
    RESOURCE_FILES([PREFIX {prefix}] {path}) expands into
    RESOURCE({path} resfs/file/{prefix}{path}
        - resfs/src/resfs/file/{prefix}{path}={rootrel_arc_src(path)}
    )

    resfs/src/{key} stores a source root (or build root) relative path of the
    source of the value of the {key} resource.

    resfs/file/{key} stores any value whose source was a file on a filesystem.
    resfs/src/resfs/file/{key} must store its path.

    This form is for use from other plugins:
    RESOURCE_FILES([DEST {dest}] {path}) expands into RESOURCE({path} resfs/file/{dest})
    """
    prefix = ''
    dest = None
    res = []

    args = iter(args)
    for arg in args:
        if arg == 'PREFIX':
            prefix, dest = next(args), None
        elif arg == 'DEST':
            dest, prefix = next(args), None
        else:
            path = arg
            key = 'resfs/file/' + (dest or (prefix + path))
            src = 'resfs/src/{}={}'.format(key, rootrel_arc_src(path, unit))
            res += ['-', src, path, key]

    unit.onresource(res)
