

PY3_LIBRARY()

LICENSE(BSD-3-Clause)

PROVIDES(numpy)

VERSION(1.23.5)

NO_COMPILER_WARNINGS()
NO_EXTENDED_SOURCE_SEARCH()

PEERDIR(
    contrib/libs/clapack
    contrib/python/numpy/py3/numpy/random
)

ADDINCL(
    contrib/python/numpy/include/numpy/core
    contrib/python/numpy/include/numpy/core/include
    FOR cython contrib/python/numpy/include/numpy/core/include
    contrib/python/numpy/include/numpy/core/include/numpy
    contrib/python/numpy/include/numpy/core/src
    contrib/python/numpy/include/numpy/core/src/common
    contrib/python/numpy/include/numpy/core/src/multiarray
    contrib/python/numpy/include/numpy/core/src/npymath
    contrib/python/numpy/include/numpy/core/src/npysort
    contrib/python/numpy/include/numpy/core/src/umath
    contrib/python/numpy/include/numpy/distutils/include
)

CFLAGS(
    -DHAVE_CBLAS
    -DHAVE_NPY_CONFIG_H=1
    -DNO_ATLAS_INFO=1
    -D_FILE_OFFSET_BITS=64
    -D_LARGEFILE64_SOURCE=1
    -D_LARGEFILE_SOURCE=1
    -DNPY_INTERNAL_BUILD=1
)

IF (ARCH_PPC64LE)
    CFLAGS(-DNPY_DISABLE_OPTIMIZATION=1)
ENDIF()

IF (CLANG)
    CFLAGS(
        -ffp-exception-behavior=strict
    )
ENDIF()

NO_CHECK_IMPORTS(
    numpy._pyinstaller.*
    numpy.distutils.command.*
    numpy.distutils.msvc9compiler
    numpy.testing._private.noseclasses
    numpy.typing._extended_precision
)

NO_LINT()

PY_SRCS(
    TOP_LEVEL
    numpy/__config__.py
    numpy/__init__.py
    numpy/__init__.pyi
    numpy/_distributor_init.py
    numpy/_globals.py
    numpy/_pyinstaller/__init__.py
    numpy/_pyinstaller/hook-numpy.py
    numpy/_pyinstaller/pyinstaller-smoke.py
    numpy/_pytesttester.py
    numpy/_pytesttester.pyi
    numpy/_typing/__init__.py
    numpy/_typing/_add_docstring.py
    numpy/_typing/_array_like.py
    numpy/_typing/_callable.pyi
    numpy/_typing/_char_codes.py
    numpy/_typing/_dtype_like.py
    numpy/_typing/_extended_precision.py
    numpy/_typing/_generic_alias.py
    numpy/_typing/_nbit.py
    numpy/_typing/_nested_sequence.py
    numpy/_typing/_scalars.py
    numpy/_typing/_shape.py
    numpy/_typing/_ufunc.pyi
    numpy/_version.py
    numpy/array_api/__init__.py
    numpy/array_api/_array_object.py
    numpy/array_api/_constants.py
    numpy/array_api/_creation_functions.py
    numpy/array_api/_data_type_functions.py
    numpy/array_api/_dtypes.py
    numpy/array_api/_elementwise_functions.py
    numpy/array_api/_manipulation_functions.py
    numpy/array_api/_searching_functions.py
    numpy/array_api/_set_functions.py
    numpy/array_api/_sorting_functions.py
    numpy/array_api/_statistical_functions.py
    numpy/array_api/_typing.py
    numpy/array_api/_utility_functions.py
    numpy/array_api/linalg.py
    numpy/compat/__init__.py
    numpy/compat/_inspect.py
    numpy/compat/_pep440.py
    numpy/compat/py3k.py
    numpy/core/__init__.py
    numpy/core/__init__.pyi
    numpy/core/_add_newdocs.py
    numpy/core/_add_newdocs_scalars.py
    numpy/core/_asarray.py
    numpy/core/_asarray.pyi
    numpy/core/_dtype.py
    numpy/core/_dtype_ctypes.py
    numpy/core/_exceptions.py
    numpy/core/_internal.py
    numpy/core/_internal.pyi
    numpy/core/_machar.py
    numpy/core/_methods.py
    numpy/core/_string_helpers.py
    numpy/core/_type_aliases.py
    numpy/core/_type_aliases.pyi
    numpy/core/_ufunc_config.py
    numpy/core/_ufunc_config.pyi
    numpy/core/arrayprint.py
    numpy/core/arrayprint.pyi
    numpy/core/defchararray.py
    numpy/core/defchararray.pyi
    numpy/core/einsumfunc.py
    numpy/core/einsumfunc.pyi
    numpy/core/fromnumeric.py
    numpy/core/fromnumeric.pyi
    numpy/core/function_base.py
    numpy/core/function_base.pyi
    numpy/core/getlimits.py
    numpy/core/getlimits.pyi
    numpy/core/memmap.py
    numpy/core/memmap.pyi
    numpy/core/multiarray.py
    numpy/core/multiarray.pyi
    numpy/core/numeric.py
    numpy/core/numeric.pyi
    numpy/core/numerictypes.py
    numpy/core/numerictypes.pyi
    numpy/core/overrides.py
    numpy/core/records.py
    numpy/core/records.pyi
    numpy/core/shape_base.py
    numpy/core/shape_base.pyi
    numpy/core/umath.py
    numpy/core/umath_tests.py
    numpy/ctypeslib.py
    numpy/ctypeslib.pyi
    numpy/distutils/__config__.py
    numpy/distutils/__init__.py
    numpy/distutils/__init__.pyi
    numpy/distutils/_shell_utils.py
    numpy/distutils/armccompiler.py
    numpy/distutils/ccompiler.py
    numpy/distutils/ccompiler_opt.py
    numpy/distutils/command/__init__.py
    numpy/distutils/command/autodist.py
    numpy/distutils/command/bdist_rpm.py
    numpy/distutils/command/build.py
    numpy/distutils/command/build_clib.py
    numpy/distutils/command/build_ext.py
    numpy/distutils/command/build_py.py
    numpy/distutils/command/build_scripts.py
    numpy/distutils/command/build_src.py
    numpy/distutils/command/config.py
    numpy/distutils/command/config_compiler.py
    numpy/distutils/command/develop.py
    numpy/distutils/command/egg_info.py
    numpy/distutils/command/install.py
    numpy/distutils/command/install_clib.py
    numpy/distutils/command/install_data.py
    numpy/distutils/command/install_headers.py
    numpy/distutils/command/sdist.py
    numpy/distutils/conv_template.py
    numpy/distutils/core.py
    numpy/distutils/cpuinfo.py
    numpy/distutils/exec_command.py
    numpy/distutils/extension.py
    numpy/distutils/fcompiler/__init__.py
    numpy/distutils/fcompiler/absoft.py
    numpy/distutils/fcompiler/arm.py
    numpy/distutils/fcompiler/compaq.py
    numpy/distutils/fcompiler/environment.py
    numpy/distutils/fcompiler/fujitsu.py
    numpy/distutils/fcompiler/g95.py
    numpy/distutils/fcompiler/gnu.py
    numpy/distutils/fcompiler/hpux.py
    numpy/distutils/fcompiler/ibm.py
    numpy/distutils/fcompiler/intel.py
    numpy/distutils/fcompiler/lahey.py
    numpy/distutils/fcompiler/mips.py
    numpy/distutils/fcompiler/nag.py
    numpy/distutils/fcompiler/none.py
    numpy/distutils/fcompiler/nv.py
    numpy/distutils/fcompiler/pathf95.py
    numpy/distutils/fcompiler/pg.py
    numpy/distutils/fcompiler/sun.py
    numpy/distutils/fcompiler/vast.py
    numpy/distutils/from_template.py
    numpy/distutils/intelccompiler.py
    numpy/distutils/lib2def.py
    numpy/distutils/line_endings.py
    numpy/distutils/log.py
    numpy/distutils/mingw32ccompiler.py
    numpy/distutils/misc_util.py
    numpy/distutils/msvc9compiler.py
    numpy/distutils/msvccompiler.py
    numpy/distutils/npy_pkg_config.py
    numpy/distutils/numpy_distribution.py
    numpy/distutils/pathccompiler.py
    numpy/distutils/system_info.py
    numpy/distutils/unixccompiler.py
    numpy/doc/__init__.py
    numpy/doc/constants.py
    numpy/doc/ufuncs.py
    numpy/dual.py
    numpy/fft/__init__.py
    numpy/fft/__init__.pyi
    numpy/fft/_pocketfft.py
    numpy/fft/_pocketfft.pyi
    numpy/fft/helper.py
    numpy/fft/helper.pyi
    numpy/lib/__init__.py
    numpy/lib/__init__.pyi
    numpy/lib/_datasource.py
    numpy/lib/_iotools.py
    numpy/lib/_version.py
    numpy/lib/_version.pyi
    numpy/lib/arraypad.py
    numpy/lib/arraypad.pyi
    numpy/lib/arraysetops.py
    numpy/lib/arraysetops.pyi
    numpy/lib/arrayterator.py
    numpy/lib/arrayterator.pyi
    numpy/lib/format.py
    numpy/lib/format.pyi
    numpy/lib/function_base.py
    numpy/lib/function_base.pyi
    numpy/lib/histograms.py
    numpy/lib/histograms.pyi
    numpy/lib/index_tricks.py
    numpy/lib/index_tricks.pyi
    numpy/lib/mixins.py
    numpy/lib/mixins.pyi
    numpy/lib/nanfunctions.py
    numpy/lib/nanfunctions.pyi
    numpy/lib/npyio.py
    numpy/lib/npyio.pyi
    numpy/lib/polynomial.py
    numpy/lib/polynomial.pyi
    numpy/lib/recfunctions.py
    numpy/lib/scimath.py
    numpy/lib/scimath.pyi
    numpy/lib/shape_base.py
    numpy/lib/shape_base.pyi
    numpy/lib/stride_tricks.py
    numpy/lib/stride_tricks.pyi
    numpy/lib/twodim_base.py
    numpy/lib/twodim_base.pyi
    numpy/lib/type_check.py
    numpy/lib/type_check.pyi
    numpy/lib/ufunclike.py
    numpy/lib/ufunclike.pyi
    numpy/lib/user_array.py
    numpy/lib/utils.py
    numpy/lib/utils.pyi
    numpy/linalg/__init__.py
    numpy/linalg/__init__.pyi
    numpy/linalg/linalg.py
    numpy/linalg/linalg.pyi
    numpy/ma/__init__.py
    numpy/ma/__init__.pyi
    numpy/ma/bench.py
    numpy/ma/core.py
    numpy/ma/core.pyi
    numpy/ma/extras.py
    numpy/ma/extras.pyi
    numpy/ma/mrecords.py
    numpy/ma/mrecords.pyi
    numpy/ma/testutils.py
    numpy/ma/timer_comparison.py
    numpy/matlib.py
    numpy/matrixlib/__init__.py
    numpy/matrixlib/__init__.pyi
    numpy/matrixlib/defmatrix.py
    numpy/matrixlib/defmatrix.pyi
    numpy/polynomial/__init__.py
    numpy/polynomial/__init__.pyi
    numpy/polynomial/_polybase.py
    numpy/polynomial/_polybase.pyi
    numpy/polynomial/chebyshev.py
    numpy/polynomial/chebyshev.pyi
    numpy/polynomial/hermite.py
    numpy/polynomial/hermite.pyi
    numpy/polynomial/hermite_e.py
    numpy/polynomial/hermite_e.pyi
    numpy/polynomial/laguerre.py
    numpy/polynomial/laguerre.pyi
    numpy/polynomial/legendre.py
    numpy/polynomial/legendre.pyi
    numpy/polynomial/polynomial.py
    numpy/polynomial/polynomial.pyi
    numpy/polynomial/polyutils.py
    numpy/polynomial/polyutils.pyi
    numpy/testing/__init__.py
    numpy/testing/__init__.pyi
    numpy/testing/_private/__init__.py
    numpy/testing/_private/decorators.py
    numpy/testing/_private/extbuild.py
    numpy/testing/_private/noseclasses.py
    numpy/testing/_private/nosetester.py
    numpy/testing/_private/parameterized.py
    numpy/testing/_private/utils.py
    numpy/testing/_private/utils.pyi
    numpy/testing/print_coercion_tables.py
    numpy/testing/utils.py
    numpy/typing/__init__.py
    numpy/typing/mypy_plugin.py
    numpy/version.py
)

SRCS(
    numpy/core/src/_simd/_simd.c
    numpy/core/src/common/array_assign.c
    numpy/core/src/common/cblasfuncs.c
    numpy/core/src/common/mem_overlap.c
    numpy/core/src/common/npy_argparse.c
    numpy/core/src/common/npy_cpu_features.c
    numpy/core/src/common/npy_hashtable.c
    numpy/core/src/common/npy_longdouble.c
    numpy/core/src/common/numpyos.c
    # numpy/core/src/common/python_xerbla.c is defined in blas.
    numpy/core/src/common/ucsnarrow.c
    numpy/core/src/common/ufunc_override.c
    numpy/core/src/dummymodule.c
    numpy/core/src/multiarray/_multiarray_tests.c
    numpy/core/src/multiarray/abstractdtypes.c
    numpy/core/src/multiarray/alloc.c
    numpy/core/src/multiarray/array_assign_array.c
    numpy/core/src/multiarray/array_assign_scalar.c
    numpy/core/src/multiarray/array_coercion.c
    numpy/core/src/multiarray/array_method.c
    numpy/core/src/multiarray/arrayfunction_override.c
    numpy/core/src/multiarray/arrayobject.c
    numpy/core/src/multiarray/arraytypes.c
    numpy/core/src/multiarray/buffer.c
    numpy/core/src/multiarray/calculation.c
    numpy/core/src/multiarray/common.c
    numpy/core/src/multiarray/common_dtype.c
    numpy/core/src/multiarray/compiled_base.c
    numpy/core/src/multiarray/conversion_utils.c
    numpy/core/src/multiarray/convert.c
    numpy/core/src/multiarray/convert_datatype.c
    numpy/core/src/multiarray/ctors.c
    numpy/core/src/multiarray/datetime.c
    numpy/core/src/multiarray/datetime_busday.c
    numpy/core/src/multiarray/datetime_busdaycal.c
    numpy/core/src/multiarray/datetime_strings.c
    numpy/core/src/multiarray/descriptor.c
    numpy/core/src/multiarray/dlpack.c
    numpy/core/src/multiarray/dragon4.c
    numpy/core/src/multiarray/dtype_transfer.c
    numpy/core/src/multiarray/dtypemeta.c
    numpy/core/src/multiarray/einsum.c
    numpy/core/src/multiarray/einsum_sumprod.c
    numpy/core/src/multiarray/experimental_public_dtype_api.c
    numpy/core/src/multiarray/flagsobject.c
    numpy/core/src/multiarray/getset.c
    numpy/core/src/multiarray/hashdescr.c
    numpy/core/src/multiarray/item_selection.c
    numpy/core/src/multiarray/iterators.c
    numpy/core/src/multiarray/legacy_dtype_implementation.c
    numpy/core/src/multiarray/lowlevel_strided_loops.c
    numpy/core/src/multiarray/mapping.c
    numpy/core/src/multiarray/methods.c
    numpy/core/src/multiarray/multiarraymodule.c
    numpy/core/src/multiarray/nditer_api.c
    numpy/core/src/multiarray/nditer_constr.c
    numpy/core/src/multiarray/nditer_pywrap.c
    numpy/core/src/multiarray/nditer_templ.c
    numpy/core/src/multiarray/number.c
    numpy/core/src/multiarray/refcount.c
    numpy/core/src/multiarray/scalarapi.c
    numpy/core/src/multiarray/scalartypes.c
    numpy/core/src/multiarray/sequence.c
    numpy/core/src/multiarray/shape.c
    numpy/core/src/multiarray/strfuncs.c
    numpy/core/src/multiarray/temp_elide.c
    numpy/core/src/multiarray/textreading/conversions.c
    numpy/core/src/multiarray/textreading/field_types.c
    numpy/core/src/multiarray/textreading/growth.c
    numpy/core/src/multiarray/textreading/readtext.c
    numpy/core/src/multiarray/textreading/rows.c
    numpy/core/src/multiarray/textreading/str_to_int.c
    numpy/core/src/multiarray/textreading/stream_pyobject.c
    numpy/core/src/multiarray/textreading/tokenize.cpp
    numpy/core/src/multiarray/typeinfo.c
    numpy/core/src/multiarray/usertypes.c
    numpy/core/src/multiarray/vdot.c
    numpy/core/src/npymath/_signbit.c
    numpy/core/src/npymath/halffloat.c
    numpy/core/src/npymath/ieee754.c
    numpy/core/src/npymath/ieee754.cpp
    numpy/core/src/npymath/npy_math.c
    numpy/core/src/npymath/npy_math_complex.c
    numpy/core/src/npysort/binsearch.cpp
    numpy/core/src/npysort/heapsort.cpp
    numpy/core/src/npysort/mergesort.cpp
    numpy/core/src/npysort/quicksort.cpp
    numpy/core/src/npysort/radixsort.cpp
    numpy/core/src/npysort/selection.cpp
    numpy/core/src/npysort/timsort.cpp
    numpy/core/src/umath/_operand_flag_tests.c
    numpy/core/src/umath/_rational_tests.c
    numpy/core/src/umath/_scaled_float_dtype.c
    numpy/core/src/umath/_struct_ufunc_tests.c
    numpy/core/src/umath/_umath_tests.c
    numpy/core/src/umath/clip.cpp
    numpy/core/src/umath/dispatching.c
    numpy/core/src/umath/extobj.c
    numpy/core/src/umath/legacy_array_method.c
    numpy/core/src/umath/loops.c
    numpy/core/src/umath/matmul.c
    numpy/core/src/umath/override.c
    numpy/core/src/umath/reduction.c
    numpy/core/src/umath/scalarmath.c
    numpy/core/src/umath/ufunc_object.c
    numpy/core/src/umath/ufunc_type_resolution.c
    numpy/core/src/umath/umathmodule.c
    numpy/core/src/umath/wrapping_array_method.c
    numpy/f2py/src/fortranobject.c
    numpy/fft/_pocketfft.c
    numpy/linalg/lapack_litemodule.c
    numpy/linalg/umath_linalg.cpp
)

IF (CLANG OR CLANG_CL)
    SET(F16C_FLAGS -mf16c)
ELSE()
    SET(F16C_FLAGS)
ENDIF()

SRCS(
    numpy/core/src/_simd/_simd.dispatch.c
    numpy/core/src/multiarray/argfunc.dispatch.c
    numpy/core/src/npysort/x86-qsort.dispatch.cpp
    numpy/core/src/umath/_umath_tests.dispatch.c
    numpy/core/src/umath/loops_arithm_fp.dispatch.c
    numpy/core/src/umath/loops_arithmetic.dispatch.c
    numpy/core/src/umath/loops_exponent_log.dispatch.c
    numpy/core/src/umath/loops_hyperbolic.dispatch.c
    numpy/core/src/umath/loops_minmax.dispatch.c
    numpy/core/src/umath/loops_modulo.dispatch.c
    numpy/core/src/umath/loops_trigonometric.dispatch.c
    numpy/core/src/umath/loops_umath_fp.dispatch.c
    numpy/core/src/umath/loops_unary_fp.dispatch.c
)

IF (ARCH_X86_64)
    SRC(numpy/core/src/_simd/_simd.dispatch.avx512_skx.c $AVX_CFLAGS $F16C_FLAGS $AVX2_CFLAGS $AVX512_CFLAGS)
    SRC(numpy/core/src/_simd/_simd.dispatch.avx512f.c $AVX_CFLAGS $F16C_FLAGS $AVX2_CFLAGS $AVX512_CFLAGS)
    SRC_C_AVX2(numpy/core/src/_simd/_simd.dispatch.fma3.avx2.c $F16C_FLAGS)
    SRC(numpy/core/src/_simd/_simd.dispatch.sse42.c)
    SRC_C_AVX2(numpy/core/src/multiarray/argfunc.dispatch.avx2.c $F16C_FLAGS)
    SRC(numpy/core/src/multiarray/argfunc.dispatch.avx512_skx.c $AVX_CFLAGS $F16C_FLAGS $AVX2_CFLAGS $AVX512_CFLAGS)
    SRC(numpy/core/src/multiarray/argfunc.dispatch.sse42.c)
    SRC(numpy/core/src/npysort/x86-qsort.dispatch.avx512_skx.cpp $AVX_CFLAGS $F16C_FLAGS $AVX2_CFLAGS $AVX512_CFLAGS)
    SRC_C_AVX2(numpy/core/src/umath/_umath_tests.dispatch.avx2.c $F16C_FLAGS)
    SRC(numpy/core/src/umath/_umath_tests.dispatch.sse41.c)
    SRC_C_AVX2(numpy/core/src/umath/loops_arithm_fp.dispatch.avx2.c $F16C_FLAGS)
    SRC(numpy/core/src/umath/loops_arithm_fp.dispatch.avx512f.c $AVX_CFLAGS $F16C_FLAGS $AVX2_CFLAGS $AVX512_CFLAGS)
    SRC_C_AVX2(numpy/core/src/umath/loops_arithmetic.dispatch.avx2.c $F16C_FLAGS)
    SRC(numpy/core/src/umath/loops_arithmetic.dispatch.avx512_skx.c $AVX_CFLAGS $F16C_FLAGS $AVX2_CFLAGS $AVX512_CFLAGS)
    SRC(numpy/core/src/umath/loops_arithmetic.dispatch.avx512f.c $AVX_CFLAGS $F16C_FLAGS $AVX2_CFLAGS $AVX512_CFLAGS)
    SRC(numpy/core/src/umath/loops_arithmetic.dispatch.sse41.c)
    SRC(numpy/core/src/umath/loops_exponent_log.dispatch.avx512_skx.c $AVX_CFLAGS $F16C_FLAGS $AVX2_CFLAGS $AVX512_CFLAGS)
    SRC(numpy/core/src/umath/loops_exponent_log.dispatch.avx512f.c $AVX_CFLAGS $F16C_FLAGS $AVX2_CFLAGS $AVX512_CFLAGS)
    SRC_C_AVX2(numpy/core/src/umath/loops_exponent_log.dispatch.fma3.avx2.c $F16C_FLAGS)
    SRC(numpy/core/src/umath/loops_hyperbolic.dispatch.avx512_skx.c $AVX_CFLAGS $F16C_FLAGS $AVX2_CFLAGS $AVX512_CFLAGS)
    SRC_C_AVX2(numpy/core/src/umath/loops_hyperbolic.dispatch.fma3.avx2.c $F16C_FLAGS)
    SRC_C_AVX2(numpy/core/src/umath/loops_minmax.dispatch.avx2.c $F16C_FLAGS)
    SRC(numpy/core/src/umath/loops_minmax.dispatch.avx512_skx.c $AVX_CFLAGS $F16C_FLAGS $AVX2_CFLAGS $AVX512_CFLAGS)
    SRC(numpy/core/src/umath/loops_trigonometric.dispatch.avx512f.c $AVX_CFLAGS $F16C_FLAGS $AVX2_CFLAGS $AVX512_CFLAGS)
    SRC_C_AVX2(numpy/core/src/umath/loops_trigonometric.dispatch.fma3.avx2.c $F16C_FLAGS)
    SRC(numpy/core/src/umath/loops_umath_fp.dispatch.avx512_skx.c $AVX_CFLAGS $F16C_FLAGS $AVX2_CFLAGS $AVX512_CFLAGS)
    SRC(numpy/core/src/umath/loops_unary_fp.dispatch.sse41.c)
ELSEIF (ARCH_ARM64)
    SRC(numpy/core/src/umath/_umath_tests.dispatch.asimdhp.c)
ENDIF()

PY_REGISTER(
    numpy.core._multiarray_tests
    numpy.core._multiarray_umath
    numpy.core._operand_flag_tests
    numpy.core._rational_tests
    numpy.core._simd
    numpy.core._struct_ufunc_tests
    numpy.core._umath_tests
    numpy.fft._pocketfft_internal
    numpy.linalg._umath_linalg
    numpy.linalg.lapack_lite
)

RESOURCE_FILES(
    PREFIX contrib/python/numpy/py3/
    .dist-info/METADATA
    .dist-info/entry_points.txt
    .dist-info/top_level.txt
    numpy/py.typed
)

END()

RECURSE(
    numpy/f2py
    numpy/random
)

RECURSE_FOR_TESTS(
    tests
)
