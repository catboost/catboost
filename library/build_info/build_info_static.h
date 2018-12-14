#pragma once

#if defined(__cplusplus)
extern "C" {
#endif

const char* GetCompilerVersion();
const char* GetCompilerFlags(); // "-std=c++14 -DNDEBUG -O2 -m64 ..."
const char* GetBuildInfo();     // Compiler version and flags

#if defined(__cplusplus)
}
#endif
