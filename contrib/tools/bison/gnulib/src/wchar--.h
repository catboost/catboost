#pragma once

#include <wchar.h>

#if defined(_WIN32)
int wcwidth(wchar_t c);
#endif
