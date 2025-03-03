#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#if defined(__clang__)
#pragma GCC diagnostic ignored "-Wunknown-warning-option"
#endif
#pragma GCC diagnostic ignored "-Wsubobject-linkage"
#endif

#include "util/memory/addstorage.cpp"
#include "util/memory/alloc.cpp"
#include "util/memory/blob.cpp"
#include "util/memory/mmapalloc.cpp"
#include "util/memory/pool.cpp"
#include "util/memory/segmented_string_pool.cpp"
#include "util/memory/segpool_alloc.cpp"
#include "util/memory/smallobj.cpp"
#include "util/memory/tempbuf.cpp"
