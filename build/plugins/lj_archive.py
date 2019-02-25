def onlj_archive(unit, *args):
    def iter_luas(l):
        for a in l:
            if a.endswith('.lua'):
                yield a

    def iter_objs(l):
        for a in l:
            s = a[:-3] + 'raw'
            unit.onluajit_objdump(['OUT', s, a])
            yield s

    luas = list(iter_luas(args))
    objs = list(iter_objs(luas))

    unit.onarchive_by_keys(['DONTCOMPRESS', 'NAME', 'LuaScripts.inc', 'KEYS', ':'.join(luas)] + objs)
    unit.onarchive_by_keys(['DONTCOMPRESS', 'NAME', 'LuaSources.inc', 'KEYS', ':'.join(luas)] + luas)

def onlj_21_archive(unit, *args):
    def iter_luas(l):
        for a in l:
            if a.endswith('.lua'):
                yield a

    def iter_objs(l):
        for a in l:
            s = a[:-3] + 'raw'
            unit.onluajit_21_objdump(['OUT', s, a])
            yield s

    luas = list(iter_luas(args))
    objs = list(iter_objs(luas))

    unit.onarchive_by_keys(['DONTCOMPRESS', 'NAME', 'LuaScripts.inc', 'KEYS', ':'.join(luas)] + objs)
    unit.onarchive_by_keys(['DONTCOMPRESS', 'NAME', 'LuaSources.inc', 'KEYS', ':'.join(luas)] + luas)

