#include <util/system/defaults.h>

#ifdef _win_

    #include <stdio.h>
    #include "dirent_win.h"

    #if defined(_MSC_VER) && (_MSC_VER < 1900)
void __cdecl _dosmaperr(unsigned long);

static void SetErrno() {
    _dosmaperr(GetLastError());
}
    #else
void __cdecl __acrt_errno_map_os_error(unsigned long const oserrno);

static void SetErrno() {
    __acrt_errno_map_os_error(GetLastError());
}
    #endif

struct DIR* opendir(const char* dirname) {
    struct DIR* dir = (struct DIR*)malloc(sizeof(struct DIR));
    if (!dir) {
        return NULL;
    }
    dir->sh = INVALID_HANDLE_VALUE;
    dir->fff_templ = NULL;
    dir->file_no = 0;
    dir->readdir_buf = NULL;

    int len = strlen(dirname);
    // Remove trailing slashes
    while (len && (dirname[len - 1] == '\\' || dirname[len - 1] == '/')) {
        --len;
    }
    int len_converted = MultiByteToWideChar(CP_UTF8, 0, dirname, len, 0, 0);
    if (len_converted == 0) {
        closedir(dir);
        return NULL;
    }
    dir->fff_templ = (WCHAR*)malloc((len_converted + 5) * sizeof(WCHAR));
    if (!dir->fff_templ) {
        closedir(dir);
        return NULL;
    }
    MultiByteToWideChar(CP_UTF8, 0, dirname, len, dir->fff_templ, len_converted);

    WCHAR append[] = {'\\', '*', '.', '*', 0};
    memcpy(dir->fff_templ + len_converted, append, sizeof(append));
    dir->sh = FindFirstFileW(dir->fff_templ, &dir->wfd);
    if (dir->sh == INVALID_HANDLE_VALUE) {
        SetErrno();
        closedir(dir);
        return NULL;
    }

    return dir;
}

int closedir(struct DIR* dir) {
    if (dir->sh != INVALID_HANDLE_VALUE) {
        FindClose(dir->sh);
    }
    free(dir->fff_templ);
    free(dir->readdir_buf);
    free(dir);
    return 0;
}

int readdir_r(struct DIR* dir, struct dirent* entry, struct dirent** result) {
    if (!FindNextFileW(dir->sh, &dir->wfd)) {
        int err = GetLastError();
        *result = 0;
        if (err == ERROR_NO_MORE_FILES) {
            SetLastError(0);
            return 0;
        } else {
            return err;
        }
    }
    entry->d_fileno = dir->file_no++;
    entry->d_reclen = sizeof(struct dirent);
    if (dir->wfd.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT &&
        (dir->wfd.dwReserved0 == IO_REPARSE_TAG_MOUNT_POINT || dir->wfd.dwReserved0 == IO_REPARSE_TAG_SYMLINK))
    {
        entry->d_type = DT_LNK;
    } else if (dir->wfd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
        entry->d_type = DT_DIR;
    } else {
        entry->d_type = DT_REG;
    }
    int len = lstrlenW(dir->wfd.cFileName);
    int conv_len = WideCharToMultiByte(CP_UTF8, 0, dir->wfd.cFileName, len, 0, 0, 0, 0);
    if (conv_len == 0) {
        return -1;
    }
    if (conv_len > sizeof(entry->d_name) - 1) {
        SetLastError(ERROR_INSUFFICIENT_BUFFER);
        return ERROR_INSUFFICIENT_BUFFER;
    }
    entry->d_namlen = conv_len;
    WideCharToMultiByte(CP_UTF8, 0, dir->wfd.cFileName, len, entry->d_name, conv_len, 0, 0);
    entry->d_name[conv_len] = 0;
    *result = entry;
    return 0;
}

struct dirent* readdir(struct DIR* dir) {
    struct dirent* res;
    if (!dir->readdir_buf) {
        dir->readdir_buf = (struct dirent*)malloc(sizeof(struct dirent));
        if (dir->readdir_buf == 0) {
            return 0;
        }
    }
    readdir_r(dir, dir->readdir_buf, &res);
    return res;
}

void rewinddir(struct DIR* dir) {
    FindClose(dir->sh);
    dir->sh = FindFirstFileW(dir->fff_templ, &dir->wfd);
    dir->file_no = 0;
}

#endif // _win_
