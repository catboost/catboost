project('x86-simd-sort', 'cpp',
        version : '1.0.0',
        license : 'BSD 3-clause')
cpp = meson.get_compiler('cpp')
src = include_directories('src')
bench = include_directories('benchmarks')
utils = include_directories('utils')
tests = include_directories('tests')
gtest_dep = dependency('gtest_main', required : true, static: true)
gbench_dep = dependency('benchmark', required : true, static: true)

fp16code = '''#include<immintrin.h>
int main() { 
  __m512h temp = _mm512_set1_ph(1.0f);
  __m512h var2 = _mm512_min_ph(temp, temp);
  return 0;
}
'''
cancompilefp16 = cpp.compiles(fp16code, args:'-march=sapphirerapids')

subdir('utils')
subdir('tests')
subdir('benchmarks')

testexe = executable('testexe',
                     include_directories : [src, utils],
                     dependencies : gtest_dep,
                     link_whole : [libtests, libcpuinfo]
                    )

benchexe = executable('benchexe',
                      include_directories : [src, utils, bench],
                      dependencies : [gbench_dep],
                      link_args: ['-lbenchmark_main'],
                      link_whole : [libbench, libcpuinfo],
                     )

summary({
  'Can compile AVX-512 FP16 ISA': cancompilefp16,
  },
  section: 'Configuration',
  bool_yn: true
  )

