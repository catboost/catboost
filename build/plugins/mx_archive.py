def onmx_formulas(unit, *args):
    """
         @usage: MX_FORMULAS(BinFiles...) # deprecated, matrixnet
         Create MatrixNet formulas archive
    """
    def iter_infos():
        for a in args:
            if a.endswith('.bin'):
                unit.on_mx_bin_to_info([a])
                yield a[:-3] + 'info'
            else:
                yield a

    infos = list(iter_infos())
    unit.onarchive_asm(['NAME', 'MxFormulas'] + infos)
    unit.on_mx_gen_table(infos)
