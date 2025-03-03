#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#if defined(__clang__)
#pragma GCC diagnostic ignored "-Wunknown-warning-option"
#endif
#pragma GCC diagnostic ignored "-Wsubobject-linkage"
#endif

#include "util/thread/factory.cpp"
#include "util/thread/fwd.cpp"
#include "util/thread/lfqueue.cpp"
#include "util/thread/lfstack.cpp"
#include "util/thread/pool.cpp"
#include "util/thread/singleton.cpp"
