#include "getpid.h"

#ifdef _win_
    // The include file should be Windows.h for Windows <=7, Processthreadsapi.h for Windows >=8 and Server 2012,
    // see http://msdn.microsoft.com/en-us/library/windows/desktop/ms683180%28v=vs.85%29.aspx
    // The way to determine windows version is described in http://msdn.microsoft.com/en-us/library/windows/desktop/aa383745%28v=vs.85%29.aspx
    // with additions about Windows Server 2012 in https://social.msdn.microsoft.com/forums/vstudio/en-US/8d76d1d7-d078-4c55-963b-77e060845d0c/what-is-ntddiversion-value-for-ws-2012
    #include <Windows.h>
    #if defined(NTDDI_WIN8) && (NTDDI_VERSION >= NTDDI_WIN8)
        #include <processthreadsapi.h>
    #endif
#else
    #include <unistd.h>
#endif

TProcessId GetPID() {
#ifdef _win_
    return GetCurrentProcessId();
#else
    return getpid();
#endif
}
