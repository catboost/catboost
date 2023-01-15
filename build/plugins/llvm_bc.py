import sys

from _common import rootrel_arc_src, sort_by_keywords, skip_build_root, stripext


def onllvm_bc(unit, *args):
    free_args, kwds = sort_by_keywords({'SYMBOLS': -1, 'NAME': 1, 'NO_COMPILE': 0}, args)
    name = kwds['NAME'][0]
    symbols = kwds.get('SYMBOLS')
    obj_suf = unit.get('OBJ_SUF')
    skip_compile_step = 'NO_COMPILE' in kwds 
    merged_bc = name + '_merged' + obj_suf + '.bc'
    out_bc = name + '_optimized' + obj_suf + '.bc'
    bcs = []
    for x in free_args:
        rel_path = rootrel_arc_src(x, unit)
        bc_path = '${ARCADIA_BUILD_ROOT}/' + skip_build_root(rel_path) + obj_suf + '.bc'
        if not skip_compile_step:
            if x.endswith('.c'):
                llvm_compile = unit.onllvm_compile_c
            elif x.endswith('.ll'):
                llvm_compile = unit.onllvm_compile_ll
            else:
                llvm_compile = unit.onllvm_compile_cxx
            llvm_compile([rel_path, bc_path])
        bcs.append(bc_path)
    unit.onllvm_link([merged_bc] + bcs)
    opt_opts = ['-O2', '-globalopt', '-globaldce']
    if symbols:
        # XXX: '#' used instead of ',' to overcome ymake tendency to split everything by comma
        opt_opts += ['-internalize', '-internalize-public-api-list=' + '#'.join(symbols)]
    unit.onllvm_opt([merged_bc, out_bc] + opt_opts)
    unit.onresource([out_bc, '/llvm_bc/' + name])
