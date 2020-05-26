#include "build_info_static.h"

#include <library/cpp/build_info/buildinfo_data.h>

extern "C" const char* GetCompilerVersion() {
#if defined(BUILD_COMPILER_VERSION)
    return BUILD_COMPILER_VERSION;
#else
    return "";
#endif
}

extern "C" const char* GetCompilerFlags() {
#if defined(BUILD_COMPILER_FLAGS)
    return BUILD_COMPILER_FLAGS;
#else
    return "";
#endif
}

extern "C" const char* GetBuildInfo() {
#if defined(BUILD_INFO)
    return BUILD_INFO;
#else
    return "";
#endif
}
