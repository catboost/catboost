#pragma once

/*
 * WARN:
 * including this header does not make a lot of sense.
 * You should just #include all necessary headers from Windows SDK,
 * and then #include <util/system/win_undef.h> in order to undefine some common macros.
 */

#include <util/system/platform.h>

#if defined(_win_)
    #include <windows.h>
#endif

#include <util/system/win_undef.h>
