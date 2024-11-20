#include "dirut.h"
#include "iterator.h"
#include "filelist.h"
#include "fts.h"
#include "pathsplit.h"
#include "path.h"

#include <util/generic/yexception.h>
#include <util/system/compiler.h>
#include <util/system/fs.h>
#include <util/system/maxlen.h>
#include <util/system/yassert.h>

void SlashFolderLocal(TString& folder) {
    if (!folder) {
        return;
    }
#ifdef _win32_
    size_t pos;
    while ((pos = folder.find('/')) != TString::npos) {
        folder.replace(pos, 1, LOCSLASH_S);
    }
#endif
    if (folder[folder.size() - 1] != LOCSLASH_C) {
        folder.append(LOCSLASH_S);
    }
}

#ifndef _win32_

bool correctpath(TString& folder) {
    return resolvepath(folder, "/");
}

bool resolvepath(TString& folder, const TString& home) {
    Y_ASSERT(home && home.at(0) == '/');
    if (!folder) {
        return false;
    }
    // may be from windows
    char* ptr = folder.begin();
    while ((ptr = strchr(ptr, '\\')) != nullptr) {
        *ptr = '/';
    }

    if (folder.at(0) == '~') {
        if (folder.length() == 1 || folder.at(1) == '/') {
            folder = GetHomeDir() + (folder.data() + 1);
        } else {
            char* buf = (char*)alloca(folder.length() + 1);
            strcpy(buf, folder.data() + 1);
            char* p = strchr(buf, '/');
            if (p) {
                *p++ = 0;
            }
            passwd* pw = getpwnam(buf);
            if (pw) {
                folder = pw->pw_dir;
                folder += "/";
                if (p) {
                    folder += p;
                }
            } else {
                return false; // unknown user
            }
        }
    }
    int len = folder.length() + home.length() + 1;
    char* path = (char*)alloca(len);
    if (folder.at(0) != '/') {
        strcpy(path, home.data());
        strcpy(strrchr(path, '/') + 1, folder.data()); // the last char must be '/' if it's a dir
    } else {
        strcpy(path, folder.data());
    }
    len = strlen(path) + 1;
    // grabbed from url.cpp
    char* newpath = (char*)alloca(len + 2);
    const char** pp = (const char**)alloca(len * sizeof(char*));
    int i = 0;
    for (char* s = path; s;) {
        pp[i++] = s;
        s = strchr(s, '/');
        if (s) {
            *s++ = 0;
        }
    }

    for (int j = 1; j < i;) {
        const char*& p = pp[j];
        if (strcmp(p, ".") == 0 || strcmp(p, "") == 0) {
            if (j == i - 1) {
                p = "";
                break;
            } else {
                memmove(pp + j, pp + j + 1, (i - j - 1) * sizeof(p));
                --i;
            }
        } else if (strcmp(p, "..") == 0) {
            if (j == i - 1) {
                if (j == 1) {
                    p = "";
                } else {
                    --i;
                    pp[j - 1] = "";
                }
                break;
            } else {
                if (j == 1) {
                    memmove(pp + j, pp + j + 1, (i - j - 1) * sizeof(p));
                    --i;
                } else {
                    memmove(pp + j - 1, pp + j + 1, (i - j - 1) * sizeof(p));
                    i -= 2;
                    --j;
                }
            }
        } else {
            ++j;
        }
    }

    char* s = newpath;
    for (int k = 0; k < i; k++) {
        s = strchr(strcpy(s, pp[k]), 0);
        *s++ = '/';
    }
    *(--s) = 0;
    folder = newpath;
    return true;
}

#else

using dir_type = enum {
    dt_empty,
    dt_error,
    dt_up,
    dt_dir
};

// precondition:  *ptr != '\\' || *ptr == 0 (cause dt_error)
// postcondition: *ptr != '\\'
template <typename T>
static int next_dir(T*& ptr) {
    int has_blank = 0;
    int has_dot = 0;
    int has_letter = 0;
    int has_ctrl = 0;

    while (*ptr && *ptr != '\\') {
        int c = (unsigned char)*ptr++;
        switch (c) {
            case ' ':
                ++has_blank;
                break;
            case '.':
                ++has_dot;
                break;
            case '/':
            case ':':
            case '*':
            case '?':
            case '"':
            case '<':
            case '>':
            case '|':
                ++has_ctrl;
                break;
            default:
                if (c == 127 || c < ' ') {
                    ++has_ctrl;
                } else {
                    ++has_letter;
                }
        }
    }
    if (*ptr) {
        ++ptr;
    }
    if (has_ctrl) {
        return dt_error;
    }
    if (has_letter) {
        return dt_dir;
    }
    if (has_dot && has_blank) {
        return dt_error;
    }
    if (has_dot == 1) {
        return dt_empty;
    }
    if (has_dot == 2) {
        return dt_up;
    }
    return dt_error;
}

using disk_type = enum {
    dk_noflags = 0,
    dk_unc = 1,
    dk_hasdrive = 2,
    dk_fromroot = 4,
    dk_error = 8
};

// root slash (if any) - part of disk
template <typename T>
static int skip_disk(T*& ptr) {
    int result = dk_noflags;
    if (!*ptr) {
        return result;
    }
    if (ptr[0] == '\\' && ptr[1] == '\\') {
        result |= dk_unc | dk_fromroot;
        ptr += 2;
        if (next_dir(ptr) != dt_dir) {
            return dk_error; // has no host name
        }
        if (next_dir(ptr) != dt_dir) {
            return dk_error; // has no share name
        }
    } else {
        if (*ptr && *(ptr + 1) == ':') {
            result |= dk_hasdrive;
            ptr += 2;
        }
        if (*ptr == '\\' || *ptr == '/') {
            ++ptr;
            result |= dk_fromroot;
        }
    }
    return result;
}

int correctpath(char* cpath, const char* path) {
    if (!path || !*path) {
        *cpath = 0;
        return 1;
    }
    char* ptr = (char*)path;
    char* cptr = cpath;
    int counter = 0;
    while (*ptr) {
        char c = *ptr;
        if (c == '/') {
            c = '\\';
        }
        if (c == '\\') {
            ++counter;
        } else {
            counter = 0;
        }
        if (counter <= 1) {
            *cptr = c;
            ++cptr;
        }
        ++ptr;
    }
    *cptr = 0;
    // replace '/' by '\'
    int dk = skip_disk(cpath);

    if (dk == dk_error) {
        return 0;
    }

    char* ptr1 = ptr = cpath;
    int level = 0;
    while (*ptr) {
        switch (next_dir(ptr)) {
            case dt_dir:
                ++level;
                break;
            case dt_empty:
                memmove(ptr1, ptr, strlen(ptr) + 1);
                ptr = ptr1;
                break;
            case dt_up:
                --level;
                if (level >= 0) {
                    *--ptr1 = 0;
                    ptr1 = strrchr(cpath, '\\');
                    if (!ptr1) {
                        ptr1 = cpath;
                    } else {
                        ++ptr1;
                    }
                    memmove(ptr1, ptr, strlen(ptr) + 1);
                    ptr = ptr1;
                    break;
                } else if (level == -1 && (dk & dk_hasdrive)) {
                    if (*ptr && *(ptr + 1) == ':' && *(cpath - 2) == ':') {
                        memmove(cpath - 3, ptr, strlen(ptr) + 1);
                        return 1;
                    }
                }
                if (dk & dk_fromroot) {
                    return 0;
                }
                break;
            case dt_error:
            default:
                return 0;
        }
        ptr1 = ptr;
    }

    if ((ptr > cpath || ptr == cpath && dk & dk_unc) && *(ptr - 1) == '\\') {
        *(ptr - 1) = 0;
    }
    return 1;
}

static inline int normchar(unsigned char c) {
    return (c < 'a' || c > 'z') ? c : c - 32;
}

static inline char* strslashcat(char* a, const char* b) {
    size_t len = strlen(a);
    if (len && a[len - 1] != '\\') {
        a[len++] = '\\';
    }
    strcpy(a + len, b);
    return a;
}

int resolvepath(char* apath, const char* rpath, const char* cpath) {
    const char* redisk = rpath;
    if (!rpath || !*rpath) {
        return 0;
    }
    int rdt = skip_disk(redisk);
    if (rdt == dk_error) {
        return 0;
    }
    if (rdt & dk_unc || rdt & dk_hasdrive && rdt & dk_fromroot) {
        return correctpath(apath, rpath);
    }

    const char* cedisk = cpath;
    if (!cpath || !*cpath) {
        return 0;
    }
    int cdt = skip_disk(cedisk);
    if (cdt == dk_error) {
        return 0;
    }

    char* tpath = (char*)alloca(strlen(rpath) + strlen(cpath) + 3);

    // rdt&dk_hasdrive && !rdt&dk_fromroot
    if (rdt & dk_hasdrive) {
        if (!(cdt & dk_fromroot)) {
            return 0;
        }
        if (cdt & dk_hasdrive && normchar(*rpath) != normchar(*cpath)) {
            return 0;
        }
        memcpy(tpath, rpath, 2);
        memcpy(tpath + 2, cedisk, strlen(cedisk) + 1);
        strslashcat(tpath, redisk);

        // !rdt&dk_hasdrive && rdt&dk_fromroot
    } else if (rdt & dk_fromroot) {
        if (!(cdt & dk_hasdrive) && !(cdt & dk_unc)) {
            return 0;
        }
        memcpy(tpath, cpath, cedisk - cpath);
        tpath[cedisk - cpath] = 0;
        strslashcat(tpath, redisk);

        // !rdt&dk_hasdrive && !rdt&dk_fromroot
    } else {
        if (!(cdt & dk_fromroot) || !(cdt & dk_hasdrive) && !(cdt & dk_unc)) {
            return 0;
        }
        strslashcat(strcpy(tpath, cpath), redisk);
    }

    return correctpath(apath, tpath);
}

bool correctpath(TString& filename) {
    char* ptr = (char*)alloca(filename.size() + 2);
    if (correctpath(ptr, filename.data())) {
        filename = ptr;
        return true;
    }
    return false;
}

bool resolvepath(TString& folder, const TString& home) {
    char* ptr = (char*)alloca(folder.size() + 3 + home.size());
    if (resolvepath(ptr, folder.data(), home.data())) {
        folder = ptr;
        return true;
    }
    return false;
}

#endif // !defined _win32_

char GetDirectorySeparator() {
    return LOCSLASH_C;
}

const char* GetDirectorySeparatorS() {
    return LOCSLASH_S;
}

void RemoveDirWithContents(TString dirName) {
    SlashFolderLocal(dirName);

    TDirIterator dir(dirName, TDirIterator::TOptions(FTS_NOSTAT));

    for (auto it = dir.begin(); it != dir.end(); ++it) {
        switch (it->fts_info) {
            case FTS_F:
            case FTS_DEFAULT:
            case FTS_DP:
            case FTS_SL:
            case FTS_SLNONE:
                if (!NFs::Remove(it->fts_path)) {
                    ythrow TSystemError() << "error while removing " << it->fts_path;
                }
                break;
        }
    }
}

int mkpath(char* path, int mode) {
    return NFs::MakeDirectoryRecursive(path, NFs::EFilePermission(mode)) ? 0 : -1;
}

// Implementation of realpath in FreeBSD (version 9.0 and less) and GetFullPathName in Windows
// did not require last component of the file name to exist (other implementations will fail
// if it does not). Use RealLocation if that behaviour is required.
TString RealPath(const TString& path) {
    TTempBuf result;
    Y_ASSERT(result.Size() > MAX_PATH); // TMP_BUF_LEN > MAX_PATH
#ifdef _win_
    if (GetFullPathName(path.data(), result.Size(), result.Data(), nullptr) == 0)
#else
    if (realpath(path.data(), result.Data()) == nullptr)
#endif
        ythrow TFileError() << "RealPath failed \"" << path << "\"";
    return result.Data();
}

TString RealLocation(const TString& path) {
    if (NFs::Exists(path)) {
        return RealPath(path);
    }
    TString dirpath = GetDirName(path);
    if (NFs::Exists(dirpath)) {
        return RealPath(dirpath) + GetDirectorySeparatorS() + GetFileNameComponent(path.data());
    }
    ythrow TFileError() << "RealLocation failed \"" << path << "\"";
}

int MakeTempDir(char path[/*FILENAME_MAX*/], const char* prefix) {
    int ret;

    TString sysTmp;

#ifdef _win32_
    if (!prefix || *prefix == '/') {
#else
    if (!prefix) {
#endif
        sysTmp = GetSystemTempDir();
        prefix = sysTmp.data();
    }

    if ((ret = ResolvePath(prefix, nullptr, path, 1)) != 0) {
        return ret;
    }
    if (!TFileStat(path).IsDir()) {
        return ENOENT;
    }
    if ((strlcat(path, "tmpXXXXXX", FILENAME_MAX) > FILENAME_MAX - 100)) {
        return EINVAL;
    }
    if (!(mkdtemp(path))) {
        return errno ? errno : EINVAL;
    }
    strcat(path, LOCSLASH_S);
    return 0;
}

bool IsDir(const TString& path) {
    return TFileStat(path).IsDir();
}

TString GetHomeDir() {
    TString s(getenv("HOME"));
    if (!s) {
#ifndef _win32_
        passwd* pw = nullptr;
        s = getenv("USER");
        if (s) {
            pw = getpwnam(s.data());
        } else {
            pw = getpwuid(getuid());
        }
        if (pw) {
            s = pw->pw_dir;
        } else
#endif
        {
            char* cur_dir = getcwd(nullptr, 0);
            s = cur_dir;
            free(cur_dir);
        }
    }
    return s;
}

void MakeDirIfNotExist(const char* path, int mode) {
    if (!NFs::MakeDirectory(path, NFs::EFilePermission(mode)) && !NFs::Exists(path)) {
        ythrow TSystemError() << "failed to create directory " << path;
    }
}

void MakePathIfNotExist(const char* path, int mode) {
    NFs::MakeDirectoryRecursive(path, NFs::EFilePermission(mode));
    if (!NFs::Exists(path) || !TFileStat(path).IsDir()) {
        ythrow TSystemError() << "failed to create directory " << path;
    }
}

const char* GetFileNameComponent(const char* f) {
    const char* p = strrchr(f, LOCSLASH_C);
#ifdef _win_
    // "/" is also valid char separator on Windows
    const char* p2 = strrchr(f, '/');
    if (p2 > p) {
        p = p2;
    }
#endif

    if (p) {
        return p + 1;
    }

    return f;
}

TString GetSystemTempDir() {
#ifdef _win_
    char buffer[1024];
    DWORD size = GetTempPath(1024, buffer);
    if (!size) {
        ythrow TSystemError() << "failed to get system temporary directory";
    }
    return TString(buffer, size);
#else
    const char* var = "TMPDIR";
    const char* def = "/tmp";
    const char* r = getenv(var);
    const char* result = r ? r : def;
    return result[0] == '/' ? result : ResolveDir(result);
#endif
}

TString ResolveDir(const char* path) {
    return ResolvePath(path, true);
}

bool SafeResolveDir(const char* path, TString& result) {
    try {
        result = ResolvePath(path, true);
        return true;
    } catch (...) {
        return false;
    }
}

TString GetDirName(const TString& path) {
    return TFsPath(path).Dirname();
}

#ifdef _win32_

char* realpath(const char* pathname, char resolved_path[MAXPATHLEN]) {
    // partial implementation: no path existence check
    return _fullpath(resolved_path, pathname, MAXPATHLEN - 1);
}

#endif

TString GetBaseName(const TString& path) {
    return TFsPath(path).Basename();
}

static bool IsAbsolutePath(const char* str) {
    return str && TPathSplitTraitsLocal::IsAbsolutePath(TStringBuf(str, NStringPrivate::GetStringLengthWithLimit(str, 3)));
}

int ResolvePath(const char* rel, const char* abs, char res[/*MAXPATHLEN*/], bool isdir) {
    char t[MAXPATHLEN * 2 + 3];
    size_t len;

    *res = 0;
    if (!rel || !*rel) {
        return EINVAL;
    }
    if (!IsAbsolutePath(rel) && IsAbsolutePath(abs)) {
        len = strlcpy(t, abs, sizeof(t));
        if (len >= sizeof(t) - 3) {
            return EINVAL;
        }
        if (t[len - 1] != LOCSLASH_C) {
            t[len++] = LOCSLASH_C;
        }
        len += strlcpy(t + len, rel, sizeof(t) - len);
    } else {
        len = strlcpy(t, rel, sizeof(t));
    }
    if (len >= sizeof(t) - 3) {
        return EINVAL;
    }
    if (isdir && t[len - 1] != LOCSLASH_C) {
        t[len++] = LOCSLASH_C;
        t[len] = 0;
    }
    if (!realpath(t, res)) {
        if (!isdir && realpath(GetDirName(t).data(), res)) {
            len = strlen(res);
            if (res[len - 1] != LOCSLASH_C) {
                res[len++] = LOCSLASH_C;
                res[len] = 0;
            }
            strcpy(res + len, GetBaseName(t).data());
            return 0;
        }
        return errno ? errno : ENOENT;
    }
    if (isdir) {
        len = strlen(res);
        if (res[len - 1] != LOCSLASH_C) {
            res[len++] = LOCSLASH_C;
            res[len] = 0;
        }
    }
    return 0;
}

TString ResolvePath(const char* rel, const char* abs, bool isdir) {
    char buf[PATH_MAX];
    if (ResolvePath(rel, abs, buf, isdir)) {
        ythrow yexception() << "cannot resolve path: \"" << rel << "\"";
    }
    return buf;
}

TString ResolvePath(const char* path, bool isDir) {
    return ResolvePath(path, nullptr, isDir);
}

TString StripFileComponent(const TString& fileName) {
    TString dir = IsDir(fileName) ? fileName : GetDirName(fileName);
    if (!dir.empty() && dir.back() != GetDirectorySeparator()) {
        dir.append(GetDirectorySeparator());
    }
    return dir;
}
