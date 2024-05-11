from os.path import join
import os
import re
import glob
from distutils.dep_util import newer


def split_fortran_files(source_dir, subroutines=None):
    """Split each file in `source_dir` into separate files per subroutine.

    Parameters
    ----------
    source_dir : str
        Full path to directory in which sources to be split are located.
    subroutines : list of str, optional
        Subroutines to split. (Default: all)

    Returns
    -------
    fnames : list of str
        List of file names (not including any path) that were created
        in `source_dir`.

    Notes
    -----
    This function is useful for code that can't be compiled with g77 because of
    type casting errors which do work with gfortran.

    Created files are named: ``original_name + '_subr_i' + '.f'``, with ``i``
    starting at zero and ending at ``num_subroutines_in_file - 1``.

    """

    if subroutines is not None:
        subroutines = [x.lower() for x in subroutines]

    def split_file(fname):
        with open(fname, 'rb') as f:
            lines = f.readlines()
            subs = []
            need_split_next = True

            # find lines with SUBROUTINE statements
            for ix, line in enumerate(lines):
                m = re.match(b'^\\s+subroutine\\s+([a-z0-9_]+)\\s*\\(', line, re.I)
                if m and line[0] not in b'Cc!*':
                    if subroutines is not None:
                        subr_name = m.group(1).decode('ascii').lower()
                        subr_wanted = (subr_name in subroutines)
                    else:
                        subr_wanted = True
                    if subr_wanted or need_split_next:
                        need_split_next = subr_wanted
                        subs.append(ix)

            # check if no split needed
            if len(subs) <= 1:
                return [fname]

            # write out one file per subroutine
            new_fnames = []
            num_files = len(subs)
            for nfile in range(num_files):
                new_fname = fname[:-2] + '_subr_' + str(nfile) + '.f'
                new_fnames.append(new_fname)
                if not newer(fname, new_fname):
                    continue
                with open(new_fname, 'wb') as fn:
                    if nfile + 1 == num_files:
                        fn.writelines(lines[subs[nfile]:])
                    else:
                        fn.writelines(lines[subs[nfile]:subs[nfile+1]])

        return new_fnames

    exclude_pattern = re.compile('_subr_[0-9]')
    source_fnames = [f for f in sorted(glob.glob(os.path.join(source_dir, '*.f')))
                             if not exclude_pattern.search(os.path.basename(f))]
    fnames = []
    for source_fname in source_fnames:
        created_files = split_file(source_fname)
        if created_files is not None:
            for cfile in created_files:
                fnames.append(os.path.basename(cfile))

    return fnames


def configuration(parent_package='', top_path=None):
    from distutils.sysconfig import get_python_inc
    from numpy.distutils.system_info import get_info, numpy_info
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    from scipy._build_utils import (get_g77_abi_wrappers,
                                    gfortran_legacy_flag_hook,
                                    blas_ilp64_pre_build_hook,
                                    get_f2py_int64_options,
                                    uses_blas64)

    config = Configuration('linalg', parent_package, top_path)

    lapack_opt = get_info('lapack_opt')

    atlas_version = ([v[3:-3] for k, v in lapack_opt.get('define_macros', [])
                      if k == 'ATLAS_INFO']+[None])[0]
    if atlas_version:
        print('ATLAS version: %s' % atlas_version)

    if uses_blas64():
        lapack_ilp64_opt = get_info('lapack_ilp64_opt', 2)

    # fblas:
    sources = ['fblas.pyf.src']
    sources += get_g77_abi_wrappers(lapack_opt)
    depends = ['fblas_l?.pyf.src']

    config.add_extension('_fblas',
                         sources=sources,
                         depends=depends,
                         extra_info=lapack_opt
                         )

    if uses_blas64():
        sources = ['fblas_64.pyf.src'] + sources[1:]
        ext = config.add_extension('_fblas_64',
                                   sources=sources,
                                   depends=depends,
                                   f2py_options=get_f2py_int64_options(),
                                   extra_info=lapack_ilp64_opt)
        ext._pre_build_hook = blas_ilp64_pre_build_hook(lapack_ilp64_opt)

    # flapack:
    sources = ['flapack.pyf.src']
    sources += get_g77_abi_wrappers(lapack_opt)
    depends = ['flapack_gen.pyf.src',
               'flapack_gen_banded.pyf.src',
               'flapack_gen_tri.pyf.src',
               'flapack_pos_def.pyf.src',
               'flapack_pos_def_tri.pyf.src',
               'flapack_sym_herm.pyf.src',
               'flapack_other.pyf.src',
               'flapack_user.pyf.src']

    config.add_extension('_flapack',
                         sources=sources,
                         depends=depends,
                         extra_info=lapack_opt
                         )

    if uses_blas64():
        sources = ['flapack_64.pyf.src'] + sources[1:]
        ext = config.add_extension('_flapack_64',
                                   sources=sources,
                                   depends=depends,
                                   f2py_options=get_f2py_int64_options(),
                                   extra_info=lapack_ilp64_opt)
        ext._pre_build_hook = blas_ilp64_pre_build_hook(lapack_ilp64_opt)

    if atlas_version is not None:
        # cblas:
        config.add_extension('_cblas',
                             sources=['cblas.pyf.src'],
                             depends=['cblas.pyf.src', 'cblas_l1.pyf.src'],
                             extra_info=lapack_opt
                             )

        # clapack:
        config.add_extension('_clapack',
                             sources=['clapack.pyf.src'],
                             depends=['clapack.pyf.src'],
                             extra_info=lapack_opt
                             )

    # _flinalg:
    config.add_extension('_flinalg',
                         sources=[join('src', 'det.f'), join('src', 'lu.f')],
                         extra_info=lapack_opt
                         )

    # _interpolative:
    routines_to_split = [
        'dfftb1',
        'dfftf1',
        'dffti1',
        'dsint1',
        'dzfft1',
        'id_srand',
        'idd_copyints',
        'idd_id2svd0',
        'idd_pairsamps',
        'idd_permute',
        'idd_permuter',
        'idd_random_transf0',
        'idd_random_transf0_inv',
        'idd_random_transf_init0',
        'idd_sfft1',
        'idd_sffti1',
        'idd_subselect',
        'iddp_asvd0',
        'iddp_rsvd0',
        'iddr_asvd0',
        'iddr_rsvd0',
        'idz_estrank0',
        'idz_id2svd0',
        'idz_permute',
        'idz_permuter',
        'idz_random_transf0_inv',
        'idz_random_transf_init0',
        'idz_random_transf_init00',
        'idz_realcomp',
        'idz_realcomplex',
        'idz_reco',
        'idz_subselect',
        'idzp_aid0',
        'idzp_aid1',
        'idzp_asvd0',
        'idzp_rsvd0',
        'idzr_asvd0',
        'idzr_reco',
        'idzr_rsvd0',
        'zfftb1',
        'zfftf1',
        'zffti1',
    ]
    print('Splitting linalg.interpolative Fortran source files')
    dirname = os.path.split(os.path.abspath(__file__))[0]
    fnames = split_fortran_files(join(dirname, 'src', 'id_dist', 'src'),
                                 routines_to_split)
    fnames = [join('src', 'id_dist', 'src', f) for f in fnames]
    ext = config.add_extension('_interpolative',
                               sources=fnames + [
                                        "interpolative.pyf"],
                               extra_info=lapack_opt
                               )
    ext._pre_build_hook = gfortran_legacy_flag_hook

    # _solve_toeplitz:
    config.add_extension('_solve_toeplitz',
                         sources=[('_solve_toeplitz.c')],
                         include_dirs=[get_numpy_include_dirs()])

    # _matfuncs_sqrtm_triu:
    if int(os.environ.get('SCIPY_USE_PYTHRAN', 1)):
        import pythran
        ext = pythran.dist.PythranExtension(
            'scipy.linalg._matfuncs_sqrtm_triu',
            sources=["scipy/linalg/_matfuncs_sqrtm_triu.py"],
            config=['compiler.blas=none'])
        config.ext_modules.append(ext)
    else:
        config.add_extension('_matfuncs_sqrtm_triu',
                             sources=[('_matfuncs_sqrtm_triu.c')],
                             include_dirs=[get_numpy_include_dirs()])

    config.add_data_dir('tests')

    # Cython BLAS/LAPACK
    config.add_data_files('cython_blas.pxd')
    config.add_data_files('cython_lapack.pxd')

    sources = ['_blas_subroutine_wrappers.f', '_lapack_subroutine_wrappers.f']
    sources += get_g77_abi_wrappers(lapack_opt)
    includes = numpy_info().get_include_dirs() + [get_python_inc()]
    config.add_library('fwrappers', sources=sources, include_dirs=includes)

    config.add_extension('cython_blas',
                         sources=['cython_blas.c'],
                         depends=['cython_blas.pyx', 'cython_blas.pxd',
                                  'fortran_defs.h', '_blas_subroutines.h'],
                         include_dirs=['.'],
                         libraries=['fwrappers'],
                         extra_info=lapack_opt)

    config.add_extension('cython_lapack',
                         sources=['cython_lapack.c'],
                         depends=['cython_lapack.pyx', 'cython_lapack.pxd',
                                  'fortran_defs.h', '_lapack_subroutines.h'],
                         include_dirs=['.'],
                         libraries=['fwrappers'],
                         extra_info=lapack_opt)

    config.add_extension('_cythonized_array_utils',
                         sources=['_cythonized_array_utils.c'],
                         depends=['_cythonized_array_utils.pyx',
                                  '_cythonized_array_utils.pxd'],
                         include_dirs=['.']
                         )
    config.add_data_files('_cythonized_array_utils.pxd')

    config.add_extension('_decomp_update', sources=['_decomp_update.c'])
    config.add_extension('_decomp_lu_cython', sources=['_decomp_lu_cython.c'])
    config.add_extension('_matfuncs_expm', sources=['_matfuncs_expm.c'])

    # Add any license files
    config.add_data_files('src/id_dist/doc/doc.tex')
    config.add_data_files('src/lapack_deprecations/LICENSE')

    # Type stubs
    config.add_data_files('*.pyi')

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup

    setup(**configuration(top_path='').todict())
