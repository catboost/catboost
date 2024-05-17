#pragma once

#define THROWING

#ifdef __cplusplus
#define THROWING noexcept
extern "C" {
#endif

const char* ArcadiaRoot() THROWING;

#ifdef __cplusplus
}
#endif

