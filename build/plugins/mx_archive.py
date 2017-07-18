def onmx_formulas(unit, *args):
    def iter_infos():
        for a in args:
            if a.endswith('.bin'):
                unit.onmx_bin_to_info([a])
                yield a[:-3] + 'info'
            else:
                yield a

    infos = list(iter_infos())
    unit.onarchive_asm(['NAME', 'MxFormulas'] + infos)
    unit.onmx_gen_table(infos)
