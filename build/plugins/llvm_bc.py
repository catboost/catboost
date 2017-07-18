import sys

from _common import rootrel_arc_src, sort_by_keywords, skip_build_root, stripext


def onllvm_bc(unit, *args):
    free_args, kwds = sort_by_keywords({'SYMBOLS': -1, 'NAME': 1}, args)
    name = kwds['NAME'][0]
    symbols = kwds.get('SYMBOLS')
    merged_bc = name + '_merged.bc'
    out_bc = name + '_optimized.bc'
    bcs = []
    for x in free_args:
        rel_path = rootrel_arc_src(x, unit)
        bc_path = '${ARCADIA_BUILD_ROOT}/' + skip_build_root(rel_path) + '.bc'
        llvm_compile = unit.onllvm_compile_c if x.endswith('.c') else unit.onllvm_compile_cxx
        llvm_compile([rel_path, bc_path])
        bcs.append(bc_path)
    unit.onllvm_link([merged_bc] + bcs)
    opt_opts = ['-O2', '-globalopt', '-globaldce']
    if symbols:
        # XXX: '#' used instead of ',' to overcome ymake tendency to split everything by comma
        opt_opts += ['-internalize', '-internalize-public-api-list=' + '#'.join(symbols)]
    unit.onllvm_opt([merged_bc, out_bc] + opt_opts)
    unit.onresource([out_bc, '/llvm_bc/' + name])
