#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#if defined(__clang__)
#pragma GCC diagnostic ignored "-Wunknown-warning-option"
#endif
#pragma GCC diagnostic ignored "-Wsubobject-linkage"
#endif

#include "util/digest/fnv.cpp"
#include "util/digest/multi.cpp"
#include "util/digest/murmur.cpp"
#include "util/digest/numeric.cpp"
#include "util/digest/sequence.cpp"
