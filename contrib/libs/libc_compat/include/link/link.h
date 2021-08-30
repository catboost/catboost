#pragma once

#ifdef _MSC_VER

#ifdef _cplusplus
extern "C" {
#endif

int link(const char *oldpath, const char *newpath);

#ifdef _cplusplus
}
#endif

#endif
