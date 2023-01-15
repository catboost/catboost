from _common import sort_by_keywords


SOURCE_ROOT = '${ARCADIA_ROOT}/'
BUILD_ROOT = '${ARCADIA_BUILD_ROOT}/'
CURDIR = '${CURDIR}/'
BINDIR = '${BINDIR}/'


def oncopy_files_to_build_prefix(unit, *args):
    keywords = {'PREFIX': 1, 'GLOBAL': 0}
    # NB! keyword 'GLOBAL' is a way to skip this word from the list of files

    flat_args, spec_args = sort_by_keywords(keywords, args)
    prefix = spec_args['PREFIX'][0] if 'PREFIX' in spec_args else ''

    if len(prefix) > 0:
        build_prefix = '/'.join([BUILD_ROOT, prefix])
    else:
        build_prefix = BUILD_ROOT

    for arg in flat_args:
        if arg.startswith(build_prefix):
            # nothing to do
            pass
        elif len(prefix) > 0 and arg.startswith(BUILD_ROOT):
            unit.oncopy_file([arg, '{}/{}'.format(build_prefix, arg[len(BUILD_ROOT):])])
        elif arg.startswith(SOURCE_ROOT):
            unit.oncopy_file([arg, '{}/{}'.format(build_prefix, arg[len(SOURCE_ROOT):])])
        else:
            offset = 0
            if arg.startswith(BINDIR):
                offset = len(BINDIR)
            elif arg.startswith(CURDIR):
                offset = len(CURDIR)
            unit.oncopy_file([arg, '{}/{}/{}'.format(build_prefix, unit.get(['MODDIR']), arg[offset:])])
