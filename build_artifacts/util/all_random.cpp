#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#if defined(__clang__)
#pragma GCC diagnostic ignored "-Wunknown-warning-option"
#endif
#pragma GCC diagnostic ignored "-Wsubobject-linkage"
#endif

#include "util/random/common_ops.cpp"
#include "util/random/easy.cpp"
#include "util/random/entropy.cpp"
#include "util/random/fast.cpp"
#include "util/random/lcg_engine.cpp"
#include "util/random/mersenne32.cpp"
#include "util/random/mersenne64.cpp"
#include "util/random/mersenne.cpp"
#include "util/random/normal.cpp"
#include "util/random/shuffle.cpp"
#include "util/random/init_atfork.cpp"
