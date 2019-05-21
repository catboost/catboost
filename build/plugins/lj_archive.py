def onlj_archive(unit, *args):
    """
        @usage: LJ_ARCHIVE(NAME Name LuaFiles...)
        Precompile .lua files using LuaJIT and archive both sources and results using sources names as keys
    """
    def iter_luas(l):
        for a in l:
            if a.endswith('.lua'):
                yield a

    def iter_objs(l):
        for a in l:
            s = a[:-3] + 'raw'
            unit.on_luajit_objdump(['OUT', s, a])
            yield s

    luas = list(iter_luas(args))
    objs = list(iter_objs(luas))

    unit.onarchive_by_keys(['DONTCOMPRESS', 'NAME', 'LuaScripts.inc', 'KEYS', ':'.join(luas)] + objs)
    unit.onarchive_by_keys(['DONTCOMPRESS', 'NAME', 'LuaSources.inc', 'KEYS', ':'.join(luas)] + luas)

def onlj_21_archive(unit, *args):
    """
        @usage: LJ_21_ARCHIVE(NAME Name LuaFiles...) # deprecated
        Precompile .lua files using LuaJIT 2.1 and archive both sources and results using sources names as keys
    """
    def iter_luas(l):
        for a in l:
            if a.endswith('.lua'):
                yield a

    def iter_objs(l):
        for a in l:
            s = a[:-3] + 'raw'
            unit.on_luajit_21_objdump(['OUT', s, a])
            yield s

    luas = list(iter_luas(args))
    objs = list(iter_objs(luas))

    unit.onarchive_by_keys(['DONTCOMPRESS', 'NAME', 'LuaScripts.inc', 'KEYS', ':'.join(luas)] + objs)
    unit.onarchive_by_keys(['DONTCOMPRESS', 'NAME', 'LuaSources.inc', 'KEYS', ':'.join(luas)] + luas)

