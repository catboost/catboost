#include "mem_info.h"

#include <util/generic/strbuf.h>
#include <util/stream/file.h>
#include <util/string/cast.h>
#include <util/string/builder.h>
#include "error.h"
#include "info.h"

#if defined(_unix_)
    #if defined(_freebsd_)
        #include <sys/sysctl.h>
        #include <sys/types.h>
        #include <sys/user.h>
    #elif defined(_darwin_) && !defined(_arm_) && !defined(__IOS__)
        #include <libproc.h>
    #elif defined(__MACH__) && defined(__APPLE__)
        #include <mach/mach.h>
    #endif
#elif defined(_win_)
    #include <Windows.h>
    #include <util/generic/ptr.h>

using NTSTATUS = LONG;
    #define STATUS_INFO_LENGTH_MISMATCH 0xC0000004
    #define STATUS_BUFFER_TOO_SMALL 0xC0000023

typedef struct _UNICODE_STRING {
    USHORT Length;
    USHORT MaximumLength;
    PWSTR Buffer;
} UNICODE_STRING, *PUNICODE_STRING;
typedef struct _CLIENT_ID {
    HANDLE UniqueProcess;
    HANDLE UniqueThread;
} CLIENT_ID, *PCLIENT_ID;
using KWAIT_REASON = ULONG;
typedef struct _SYSTEM_THREAD_INFORMATION {
    LARGE_INTEGER KernelTime;
    LARGE_INTEGER UserTime;
    LARGE_INTEGER CreateTime;
    ULONG WaitTime;
    PVOID StartAddress;
    CLIENT_ID ClientId;
    LONG Priority;
    LONG BasePriority;
    ULONG ContextSwitches;
    ULONG ThreadState;
    KWAIT_REASON WaitReason;
} SYSTEM_THREAD_INFORMATION, *PSYSTEM_THREAD_INFORMATION;
typedef struct _SYSTEM_PROCESS_INFORMATION {
    ULONG NextEntryOffset;
    ULONG NumberOfThreads;
    LARGE_INTEGER SpareLi1;
    LARGE_INTEGER SpareLi2;
    LARGE_INTEGER SpareLi3;
    LARGE_INTEGER CreateTime;
    LARGE_INTEGER UserTime;
    LARGE_INTEGER KernelTime;
    UNICODE_STRING ImageName;
    LONG BasePriority;
    HANDLE UniqueProcessId;
    HANDLE InheritedFromUniqueProcessId;
    ULONG HandleCount;
    ULONG SessionId;
    ULONG_PTR PageDirectoryBase;
    SIZE_T PeakVirtualSize;
    SIZE_T VirtualSize;
    DWORD PageFaultCount;
    SIZE_T PeakWorkingSetSize;
    SIZE_T WorkingSetSize;
    SIZE_T QuotaPeakPagedPoolUsage;
    SIZE_T QuotaPagedPoolUsage;
    SIZE_T QuotaPeakNonPagedPoolUsage;
    SIZE_T QuotaNonPagedPoolUsage;
    SIZE_T PagefileUsage;
    SIZE_T PeakPagefileUsage;
    SIZE_T PrivatePageCount;
    LARGE_INTEGER ReadOperationCount;
    LARGE_INTEGER WriteOperationCount;
    LARGE_INTEGER OtherOperationCount;
    LARGE_INTEGER ReadTransferCount;
    LARGE_INTEGER WriteTransferCount;
    LARGE_INTEGER OtherTransferCount;
    SYSTEM_THREAD_INFORMATION Threads[1];
} SYSTEM_PROCESS_INFORMATION, *PSYSTEM_PROCESS_INFORMATION;

typedef enum _SYSTEM_INFORMATION_CLASS {
    SystemBasicInformation = 0,
    SystemProcessInformation = 5,
} SYSTEM_INFORMATION_CLASS;

#else

#endif

namespace NMemInfo {
    TMemInfo GetMemInfo(pid_t pid) {
        TMemInfo result;

#if defined(_unix_)

    #if defined(_linux_) || defined(_freebsd_) || defined(_cygwin_)
        const ui32 pagesize = NSystemInfo::GetPageSize();
    #endif

    #if defined(_linux_) || defined(_cygwin_)
        TString path;
        if (!pid) {
            path = "/proc/self/statm";
        } else {
            path = TStringBuilder() << TStringBuf("/proc/") << pid << TStringBuf("/statm");
        }
        const TString stats = TUnbufferedFileInput(path).ReadAll();

        TStringBuf statsiter(stats);

        result.VMS = FromString<ui64>(statsiter.NextTok(' ')) * pagesize;
        result.RSS = FromString<ui64>(statsiter.NextTok(' ')) * pagesize;

        #if defined(_cygwin_)
        //cygwin not very accurate
        result.VMS = Max(result.VMS, result.RSS);
        #endif
    #elif defined(_freebsd_)
        int mib[4] = {CTL_KERN, KERN_PROC, KERN_PROC_PID, pid};
        size_t size = sizeof(struct kinfo_proc);

        struct kinfo_proc proc;
        Zero(proc);

        errno = 0;
        if (sysctl((int*)mib, 4, &proc, &size, nullptr, 0) == -1) {
            int err = errno;
            TString errtxt = LastSystemErrorText(err);
            ythrow yexception() << "sysctl({CTL_KERN,KERN_PROC,KERN_PROC_PID,pid},4,proc,&size,NULL,0) returned -1, errno: " << err << " (" << errtxt << ")" << Endl;
        }

        result.VMS = proc.ki_size;
        result.RSS = proc.ki_rssize * pagesize;
    #elif defined(_darwin_) && !defined(_arm_) && !defined(__IOS__)
        if (!pid) {
            pid = getpid();
        }
        struct proc_taskinfo taskInfo;
        const int r = proc_pidinfo(pid, PROC_PIDTASKINFO, 0, &taskInfo, sizeof(taskInfo));

        if (r != sizeof(taskInfo)) {
            int err = errno;
            TString errtxt = LastSystemErrorText(err);
            ythrow yexception() << "proc_pidinfo(pid, PROC_PIDTASKINFO, 0, &taskInfo, sizeof(taskInfo)) returned " << r << ", errno: " << err << " (" << errtxt << ")" << Endl;
        }
        result.VMS = taskInfo.pti_virtual_size;
        result.RSS = taskInfo.pti_resident_size;
    #elif defined(__MACH__) && defined(__APPLE__)
        Y_UNUSED(pid);
        struct mach_task_basic_info taskInfo;
        mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;

        const int r = task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&taskInfo, &infoCount);
        if (r != KERN_SUCCESS) {
            int err = errno;
            TString errtxt = LastSystemErrorText(err);
            ythrow yexception() << "task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &infoCount) returned" << r << ", errno: " << err << " (" << errtxt << ")" << Endl;
        }
        result.VMS = taskInfo.virtual_size;
        result.RSS = taskInfo.resident_size;
    #elif defined(_arm_)
        Y_UNUSED(pid);
        ythrow yexception() << "arm is not supported";
    #endif
#elif defined(_win_)
        if (!pid) {
            pid = GetCurrentProcessId();
        }

        NTSTATUS status;
        TArrayHolder<char> buffer;
        ULONG bufferSize;

        // Query data for all processes and threads in the system.
        // This is probably an overkill if the target process is normal not-privileged one,
        // but allows to obtain information even about system processes that are not open-able directly.
        typedef NTSTATUS(_stdcall * NTQSI_PROC)(SYSTEM_INFORMATION_CLASS, PVOID, ULONG, PULONG);
        NTQSI_PROC NtQuerySystemInformation = (NTQSI_PROC)GetProcAddress(GetModuleHandle(TEXT("ntdll.dll")), "NtQuerySystemInformation");
        bufferSize = 0x4000;

        for (;;) {
            buffer.Reset(new char[bufferSize]);
            status = NtQuerySystemInformation(SystemProcessInformation, buffer.Get(), bufferSize, &bufferSize);

            if (!status) {
                break;
            }

            if (status != STATUS_BUFFER_TOO_SMALL && status != STATUS_INFO_LENGTH_MISMATCH) {
                ythrow yexception() << "NtQuerySystemInformation failed with status code " << status;
            }
        }

        SYSTEM_PROCESS_INFORMATION* process = (SYSTEM_PROCESS_INFORMATION*)buffer.Get();
        while (process->UniqueProcessId != (HANDLE)(size_t)(pid)) {
            if (!process->NextEntryOffset) {
                ythrow yexception() << "GetMemInfo: invalid PID";
            }

            process = (SYSTEM_PROCESS_INFORMATION*)((char*)process + process->NextEntryOffset);
        }

        result.VMS = process->VirtualSize;
        result.RSS = process->WorkingSetSize;
#endif
        return result;
    }
}
