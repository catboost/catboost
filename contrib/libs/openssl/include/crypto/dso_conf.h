#if defined(__APPLE__) && defined(__aarch64__)
#   include "dso_conf_darwin_arm64.h"
#elif defined(__APPLE__) && defined(__IOS__)
#   include "dso_conf_darwin_arm64.h"
#elif defined(_MSC_VER) && defined(_M_X64)
#   include "dso_conf-win.h"
#elif defined(__ANDROID__)
#   include "dso_conf-android.h"
#else
#   include "dso_conf-osx-linux_aarch64-linux.h"
#endif
