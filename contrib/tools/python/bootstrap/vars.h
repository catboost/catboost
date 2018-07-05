#pragma once

#define THROWING

#ifdef __cplusplus
#define THROWING noexcept
extern "C" {
#endif

const char* GetLibDir() THROWING;
const char* GetPyLib() THROWING;

#ifdef __cplusplus
}
#endif


