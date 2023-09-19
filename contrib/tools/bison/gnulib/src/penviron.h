#pragma once

#if defined(__FreeBSD__) || defined(__MACH__)
    extern char** environ;
#endif
