/*
    Copyright (c) 2005-2021 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

// Source file for miscellaneous entities that are infrequently referenced by
// an executing program.

#include "oneapi/tbb/detail/_exception.h"
#include "oneapi/tbb/detail/_machine.h"

#include "oneapi/tbb/version.h"

#include "misc.h"
#include "governor.h"
#include "assert_impl.h" // Out-of-line TBB assertion handling routines are instantiated here.
#include "concurrent_monitor_mutex.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <cstring>
#include <cstdarg>

#if _WIN32||_WIN64
#include <windows.h>
#endif

#if !_WIN32
#include <unistd.h> // sysconf(_SC_PAGESIZE)
#endif

namespace tbb {
namespace detail {
namespace r1 {

//------------------------------------------------------------------------
// governor data
//------------------------------------------------------------------------
cpu_features_type governor::cpu_features;

//------------------------------------------------------------------------
// concurrent_monitor_mutex data
//------------------------------------------------------------------------
#if !__TBB_USE_FUTEX
std::mutex concurrent_monitor_mutex::my_init_mutex;
#endif


size_t DefaultSystemPageSize() {
#if _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwPageSize;
#else
    return sysconf(_SC_PAGESIZE);
#endif
}

/** The leading "\0" is here so that applying "strings" to the binary delivers a clean result. */
static const char VersionString[] = "\0" TBB_VERSION_STRINGS;

static bool PrintVersionFlag = false;

void PrintVersion() {
    PrintVersionFlag = true;
    std::fputs(VersionString+1,stderr);
}

void PrintExtraVersionInfo( const char* category, const char* format, ... ) {
    if( PrintVersionFlag ) {
        char str[1024]; std::memset(str, 0, 1024);
        va_list args; va_start(args, format);
        // Note: correct vsnprintf definition obtained from tbb_assert_impl.h
        std::vsnprintf( str, 1024-1, format, args);
        va_end(args);
        std::fprintf(stderr, "oneTBB: %s\t%s\n", category, str );
    }
}

//! check for transaction support.
#if _MSC_VER
#include <intrin.h> // for __cpuid
#endif

#if __TBB_x86_32 || __TBB_x86_64
void check_cpuid(int leaf, int sub_leaf, int registers[4]) {
#if _MSC_VER
    __cpuidex(registers, leaf, sub_leaf);
#else
    int reg_eax = 0;
    int reg_ebx = 0;
    int reg_ecx = 0;
    int reg_edx = 0;
#if __TBB_x86_32 && __PIC__
    // On 32-bit systems with position-independent code GCC fails to work around the stuff in EBX
    // register. We help it using backup and restore.
    __asm__("mov %%ebx, %%esi\n\t"
            "cpuid\n\t"
            "xchg %%ebx, %%esi"
            : "=a"(reg_eax), "=S"(reg_ebx), "=c"(reg_ecx), "=d"(reg_edx)
            : "0"(leaf), "2"(sub_leaf) // read value from eax and ecx
    );
#else
    __asm__("cpuid"
            : "=a"(reg_eax), "=b"(reg_ebx), "=c"(reg_ecx), "=d"(reg_edx)
            : "0"(leaf), "2"(sub_leaf) // read value from eax and ecx
    );
#endif
    registers[0] = reg_eax;
    registers[1] = reg_ebx;
    registers[2] = reg_ecx;
    registers[3] = reg_edx;
#endif
}
#endif

void detect_cpu_features(cpu_features_type& cpu_features) {
    suppress_unused_warning(cpu_features);
#if __TBB_x86_32 || __TBB_x86_64
    const int rtm_ebx_mask = 1 << 11;
    const int waitpkg_ecx_mask = 1 << 5;
    int registers[4] = {0};

    // Check RTM and WAITPKG
    check_cpuid(7, 0, registers);
    cpu_features.rtm_enabled = (registers[1] & rtm_ebx_mask) != 0;
    cpu_features.waitpkg_enabled = (registers[2] & waitpkg_ecx_mask) != 0;
#endif /* (__TBB_x86_32 || __TBB_x86_64) */
}

} // namespace r1
} // namespace detail
} // namespace tbb
