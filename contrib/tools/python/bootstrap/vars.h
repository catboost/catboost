#pragma once

#define THROWING

#ifdef __cplusplus
#define THROWING throw()
extern "C" {
#endif

const char* GetLibDir() THROWING;
const char* GetPyLib() THROWING;

#ifdef __cplusplus
}
#endif


