#include "shmat.h"

#include <util/generic/guid.h>

#if defined(_win_)
    #include <stdio.h>
    #include "winint.h"
#elif defined(_bionic_)
    #include <sys/types.h>
    #include <sys/ipc.h>
    #include <sys/syscall.h>
#elif defined(_unix_)
    #include <sys/types.h>
    #include <sys/ipc.h>
    #include <sys/shm.h>
#endif

#if defined(_cygwin_)
    #define WINAPI __stdcall
    #define FILE_MAP_ALL_ACCESS ((long)983071)
    #define PAGE_READWRITE 4
    #define FALSE 0

extern "C" {
    using HANDLE = OS_HANDLE;
    using BOOL = int;
    using DWORD = ui32;
    using LPCTSTR = const char*;
    using LPVOID = void*;
    using LPCVOID = void const*;
    using SIZE_T = size_t;

    BOOL WINAPI CloseHandle(HANDLE hObject);
    HANDLE WINAPI OpenFileMappingA(DWORD dwDesiredAccess, BOOL bInheritHandle, LPCTSTR lpName);
    LPVOID WINAPI MapViewOfFile(HANDLE hFileMappingObject, DWORD DesiredAccess, DWORD FileOffsetHigh, DWORD FileOffsetLow, SIZE_T NumberOfBytesToMap);
    HANDLE WINAPI CreateFileMappingA(HANDLE hFile, LPVOID lpAttributes, DWORD flProtect, DWORD MaximumSizeHigh, DWORD MaximumSizeLow, LPCTSTR lpName);
    BOOL WINAPI UnmapViewOfFile(LPCVOID lpBaseAddress);
    DWORD WINAPI GetLastError(void);
}
#endif

#if defined(_bionic_)
namespace {
    #if !defined(__i386__)
    static int shmget(key_t key, size_t size, int flag) {
        if (size > PTRDIFF_MAX) {
            size = SIZE_MAX;
        }

        return syscall(__NR_shmget, key, size, flag);
    }

    static void* shmat(int id, const void* addr, int flag) {
        return (void*)syscall(__NR_shmat, id, addr, flag);
    }

    static int shmctl(int id, int cmd, void* buf) {
        return syscall(__NR_shmctl, id, cmd | IPC_64, buf);
    }

    static int shmdt(const void* addr) {
        return syscall(__NR_shmdt, addr);
    }

    #else
        #define IPCOP_shmat 21
        #define IPCOP_shmdt 22
        #define IPCOP_shmget 23
        #define IPCOP_shmctl 24

    static int shmget(key_t key, size_t size, int flag) {
        return syscall(__NR_ipc, IPCOP_shmget, key, size, flag, 0);
    }

    static void* shmat(int id, const void* addr, int flag) {
        void* retval;
        long res = syscall(__NR_ipc, IPCOP_shmat, id, flag, (long)&retval, addr);
        return (res >= 0) ? retval : (void*)-1;
    }

    static int shmctl(int id, int cmd, void* buf) {
        return syscall(__NR_ipc, IPCOP_shmctl, id, cmd | IPC_64, 0, buf);
    }

    static int shmdt(const void* addr) {
        return syscall(__NR_ipc, IPCOP_shmdt, 0, 0, 0, addr);
    }
    #endif
}
#endif

TSharedMemory::TSharedMemory()
    : Handle(INVALID_FHANDLE)
    , Data(nullptr)
    , Size(0)
{
}

#if defined(_win_)
static void FormatName(char* buf, const TGUID& id) {
    sprintf(buf, "Global\\shmat-%s", GetGuidAsString(id).c_str());
}

bool TSharedMemory::Open(const TGUID& id, int size) {
    //Y_ASSERT(Data == 0);
    Id = id;
    Size = size;

    char name[100];
    FormatName(name, Id);
    Handle = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, name);

    if (Handle == 0) {
        return false;
    }

    Data = MapViewOfFile(Handle, FILE_MAP_ALL_ACCESS, 0, 0, size);

    if (Data == 0) {
        //Y_ASSERT(0);
        CloseHandle(Handle);
        Handle = INVALID_OS_HANDLE;

        return false;
    }

    return true;
}

bool TSharedMemory::Create(int size) {
    //Y_ASSERT(Data == 0);
    Size = size;

    CreateGuid(&Id);

    char name[100];
    FormatName(name, Id);
    Handle = CreateFileMappingA(INVALID_OS_HANDLE, nullptr, PAGE_READWRITE, 0, size, name);

    if (Handle == 0) {
        //Y_ASSERT(0);
        return false;
    }

    Data = MapViewOfFile(Handle, FILE_MAP_ALL_ACCESS, 0, 0, size);

    if (Data == 0) {
        //Y_ASSERT(0);
        CloseHandle(Handle);
        Handle = INVALID_OS_HANDLE;

        return false;
    }

    return true;
}

TSharedMemory::~TSharedMemory() {
    if (Data) {
        UnmapViewOfFile(Handle);
    }

    CloseHandle(Handle);
}
#else
static key_t GetKey(const TGUID& id) {
    i64 id64 = (ui64)(((ui64)id.dw[0] + (ui64)id.dw[2]) << 32) + (ui64)id.dw[1] + (ui64)id.dw[3];

    return id64;
}

bool TSharedMemory::Open(const TGUID& id, int size) {
    Y_ABORT_UNLESS(id, "invalid shared memory guid: %s", GetGuidAsString(id).data());

    //Y_ASSERT(Data == 0);
    Size = size;

    key_t k = GetKey(id);
    int shmId = shmget(k, Size, 0777); // do not fill Handle, since IPC_RMID should be called by owner

    if (shmId < 0) {
        return false;
    }

    Data = shmat(shmId, nullptr, 0);

    if (Data == nullptr) {
        //Y_ASSERT(0);
        return false;
    }

    return true;
}

bool TSharedMemory::Create(int size) {
    //Y_ASSERT(Data == 0);
    Size = size;

    CreateGuid(&Id);

    key_t k = GetKey(Id);
    Handle = shmget(k, Size, IPC_CREAT | IPC_EXCL | 0777);

    if (Handle < 0) {
        //Y_ASSERT(0);
        return false;
    }

    Data = shmat(Handle, nullptr, 0);

    if (Data == (void*)-1) {
        //Y_ASSERT(0);
        shmctl(Handle, IPC_RMID, nullptr);
        Handle = -1;

        return false;
    }

    return true;
}

TSharedMemory::~TSharedMemory() {
    if (Data) {
        shmdt(Data);
    }

    if (Handle >= 0) {
        shmctl(Handle, IPC_RMID, nullptr);
    }
}
#endif
