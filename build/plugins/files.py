def onfiles(unit, *args):
    args = list(args)
    for arg in args:
        if not arg.startswith('${ARCADIA_BUILD_ROOT}'):
            unit.oncopy_file([arg, arg])
