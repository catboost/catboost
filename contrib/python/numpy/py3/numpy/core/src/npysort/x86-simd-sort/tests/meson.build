libtests = []

if cpp.has_argument('-march=skylake-avx512')
  libtests += static_library('tests_kv',
    files(
      'test-keyvalue.cpp',
      'test-argsort.cpp',
    ),
    dependencies: gtest_dep,
    include_directories : [src, utils],
    cpp_args : ['-O3', '-march=skylake-avx512'],
    )
endif

if cpp.has_argument('-march=icelake-client')
  libtests += static_library('tests_qsort',
    files('test-qsort.cpp', ),
    dependencies: gtest_dep,
    include_directories : [src, utils],
    cpp_args : ['-O3', '-march=icelake-client'],
    )
endif

if cancompilefp16
  libtests += static_library('tests_qsortfp16',
    files('test-qsortfp16.cpp', ),
    dependencies: gtest_dep,
    include_directories : [src, utils],
    cpp_args : ['-O3', '-march=sapphirerapids'],
    )
endif
