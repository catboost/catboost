#pragma once

#include "platform.h"

#if defined(_win_)

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>

#undef GetFreeSpace
#undef LoadImage
#undef GetMessage
#undef SendMessage
#undef DeleteFile
#undef OPTIONAL
#undef GetUserName
#undef CreateMutex
#undef GetObject
#undef GetGeoInfo
#undef GetClassName
#undef LANG_LAO
#undef GetKValue
#undef StartDoc
#undef UpdateResource
#undef GetNameInfo

#undef IGNORE
#undef ERROR

#endif
