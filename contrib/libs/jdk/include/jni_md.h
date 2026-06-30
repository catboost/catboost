#pragma once

#if defined(__APPLE__)
#   include "jni_md-osx.h"
#elif defined(_MSC_VER)
#   include "jni_md-win.h"
#else
#   include "jni_md-linux.h"
#endif
