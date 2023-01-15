#pragma once

#include <cstdlib>

// http://support.microsoft.com/kb/208427
#ifndef URL_MAXLEN
    #define URL_MAXLEN 2083
#endif

#define HOST_MAX 260
#ifndef URL_MAX
    #define URL_MAX 1024
#endif
#define FULLURL_MAX (URL_MAX + HOST_MAX)

#define LINKTEXT_MAX 1024

#ifdef WIN32
    #ifndef PATH_MAX
        #define PATH_MAX _MAX_PATH
    #endif
#else

    #ifndef MAX_PATH
        #define MAX_PATH PATH_MAX
    #endif

    #ifndef _MAX_PATH
        #define _MAX_PATH PATH_MAX
    #endif

#endif
