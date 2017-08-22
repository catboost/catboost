#include "path.h"
#include "pathsplit.h"

#include <util/system/fs.h>
#include <util/system/file.h>
#include <util/system/platform.h>
#include "dirut.h"
#include <util/generic/yexception.h>

struct TFsPath::TSplit: public TAtomicRefCount<TSplit>, public TPathSplit {
    inline TSplit(const TStringBuf path)
        : TPathSplit(path)
    {
    }
};

void TFsPath::CheckDefined() const {
    if (!IsDefined()) {
        ythrow TIoException() << STRINGBUF("must be defined");
    }
}

bool TFsPath::IsSubpathOf(const TFsPath& that) const {
    const TSplit& split = GetSplit();
    const TSplit& rsplit = that.GetSplit();

    if (rsplit.IsAbsolute != split.IsAbsolute)
        return false;

    if (rsplit.Drive != split.Drive)
        return false;

    if (+rsplit >= +split)
        return false;

    return std::equal(rsplit.begin(), rsplit.end(), split.begin());
}

TFsPath TFsPath::RelativeTo(const TFsPath& root) const {
    TSplit split = GetSplit();
    const TSplit& rsplit = root.GetSplit();

    if (split.Reconstruct() == rsplit.Reconstruct())
        return TFsPath();

    if (!this->IsSubpathOf(root))
        ythrow TIoException() << "path " << *this << " is not subpath of " << root;

    split.erase(split.begin(), split.begin() + rsplit.size());
    split.IsAbsolute = false;

    return TFsPath(split.Reconstruct());
}

TFsPath TFsPath::RelativePath(const TFsPath& root) const {
    TSplit split = GetSplit();
    const TSplit& rsplit = root.GetSplit();
    size_t cnt = 0;

    while (+split > cnt && +rsplit > cnt && split[cnt] == rsplit[cnt])
        ++cnt;
    bool absboth = split.IsAbsolute && rsplit.IsAbsolute;
    if (cnt == 0 && !absboth)
        ythrow TIoException() << "No common parts in " << *this << " and " << root;
    TString r;
    for (size_t i = 0; i < +rsplit - cnt; i++)
        r += i == 0 ? ".." : "/..";
    for (size_t i = cnt; i < +split; i++) {
        r += (i == 0 || i == cnt && +rsplit - cnt == 0 ? "" : "/");
        r += split[i];
    }
    return +r ? TFsPath(r) : TFsPath();
}

TFsPath TFsPath::Parent() const {
    if (!IsDefined())
        return TFsPath();

    TSplit split = GetSplit();
    if (+split)
        split.pop_back();
    if (! + split && !split.IsAbsolute)
        return TFsPath(".");
    return TFsPath(split.Reconstruct());
}

TFsPath& TFsPath::operator/=(const TFsPath& that) {
    if (!IsDefined()) {
        *this = that;

    } else if (that.IsDefined() && that.GetPath() != ".") {
        if (!that.IsRelative())
            ythrow TIoException() << "path should be relative: " << that.GetPath();

        TSplit split = GetSplit();
        const TSplit& rsplit = that.GetSplit();
        split.insert(split.end(), rsplit.begin(), rsplit.end());
        *this = TFsPath(split.Reconstruct());
    }
    return *this;
}

TFsPath& TFsPath::Fix() {
    // just normalize via reconstruction
    TFsPath(GetSplit().Reconstruct()).Swap(*this);

    return *this;
}

TString TFsPath::GetName() const {
    if (!IsDefined())
        return TString();

    const TSplit& split = GetSplit();

    if (split.size() > 0) {
        if (split.back() != "..") {
            return TString(split.back());
        } else {
            // cannot just drop last component, because path itself may be a symlink
            return RealPath().GetName();
        }
    } else {
        if (split.IsAbsolute) {
            return split.Reconstruct();
        } else {
            return Cwd().GetName();
        }
    }
}

TString TFsPath::GetExtension() const {
    return TString(GetSplit().Extension());
}

bool TFsPath::IsAbsolute() const {
    return GetSplit().IsAbsolute;
}

bool TFsPath::IsRelative() const {
    return !IsAbsolute();
}

void TFsPath::InitSplit() const {
    Split_ = new TSplit(Path_);
}

TFsPath::TSplit& TFsPath::GetSplit() const {
    // XXX: race condition here
    if (!Split_)
        InitSplit();
    return *Split_;
}

static Y_FORCE_INLINE void VerifyPath(const TStringBuf path) {
    Y_VERIFY(!path.Contains('\0'), "wrong format of TFsPath");
}

TFsPath::TFsPath() {
}

TFsPath::TFsPath(const TString& path)
    : Path_(path)
{
    VerifyPath(Path_);
}

TFsPath::TFsPath(const TStringBuf path)
    : Path_(ToString(path))
{
    VerifyPath(Path_);
}

TFsPath::TFsPath(const TString& path, const TString& realPath)
    : Path_(path)
    , RealPath_(realPath)
{
    CheckDefined();
    Y_ASSERT(RealPath_.length() > 0);
}

TFsPath::TFsPath(const char* path)
    : Path_(path)
{
}

TFsPath TFsPath::Child(const TString& name) const {
    if (!name)
        ythrow TIoException() << "child name must not be empty";

    return *this / name;
}

struct TClosedir {
    static void Destroy(DIR* dir) {
        if (dir)
            if (0 != closedir(dir))
                ythrow TIoSystemError() << "failed to closedir";
    }
};

void TFsPath::ListNames(yvector<TString>& children) const {
    CheckDefined();
    THolder<DIR, TClosedir> dir(opendir(~*this));
    if (!dir) {
        ythrow TIoSystemError() << "failed to opendir " << Path_;
    }

    for (;;) {
        struct dirent de;
        struct dirent* ok;
        int r = readdir_r(dir.Get(), &de, &ok);
        if (r != 0)
            ythrow TIoSystemError() << "failed to readdir " << Path_;
        if (ok == nullptr)
            return;
        TString name(de.d_name);
        if (name == "." || name == "..")
            continue;
        children.push_back(name);
    }
}

void TFsPath::List(yvector<TFsPath>& files) const {
    yvector<TString> names;
    ListNames(names);
    for (auto& name : names) {
        files.push_back(Child(name));
    }
}

void TFsPath::RenameTo(const TString& newPath) const {
    CheckDefined();
    if (!newPath)
        ythrow TIoException() << "bad new file name";
    if (!NFs::Rename(Path_, newPath))
        ythrow TIoSystemError() << "failed to rename " << Path_ << " to " << newPath;
}

void TFsPath::RenameTo(const char* newPath) const {
    RenameTo(TString(newPath));
}

void TFsPath::RenameTo(const TFsPath& newPath) const {
    RenameTo(newPath.GetPath());
}

void TFsPath::Touch() const {
    CheckDefined();
    if (!TFile(*this, OpenAlways).IsOpen()) {
        ythrow TIoException() << "failed to touch " << *this;
    }
}

// XXX: move implementation to util/somewhere.
TFsPath TFsPath::RealPath() const {
    CheckDefined();
    if (RealPath_)
        return RealPath_;
    RealPath_ = ::RealPath(*this);
    return TFsPath(RealPath_, RealPath_);
}

TFsPath TFsPath::RealLocation() const {
    CheckDefined();
    if (RealPath_)
        return RealPath_;
    RealPath_ = ::RealLocation(*this);
    return TFsPath(RealPath_, RealPath_);
}

TFsPath TFsPath::ReadLink() const {
    CheckDefined();

    if (!IsSymlink())
        ythrow TIoException() << "not a symlink " << *this;

    return NFs::ReadLink(*this);
}

bool TFsPath::Exists() const {
    return IsDefined() && NFs::Exists(*this);
}

void TFsPath::CheckExists() const {
    if (!Exists()) {
        ythrow TIoException() << "path does not exist " << Path_;
    }
}

bool TFsPath::IsDirectory() const {
    return IsDefined() && TFileStat(~GetPath()).IsDir();
}

bool TFsPath::IsFile() const {
    return IsDefined() && TFileStat(~GetPath()).IsFile();
}

bool TFsPath::IsSymlink() const {
    return IsDefined() && TFileStat(~GetPath(), true).IsSymlink();
}

void TFsPath::DeleteIfExists() const {
    if (!IsDefined())
        return;

    ::unlink(~*this);
    ::rmdir(~*this);
    if (Exists()) {
        ythrow TIoException() << "failed to delete " << Path_;
    }
}

void TFsPath::MkDir(const int mode) const {
    CheckDefined();
    int r = Mkdir(~*this, mode);
    if (r != 0)
        ythrow TIoSystemError() << "could not create directory " << Path_;
}

void TFsPath::MkDirs(const int mode) const {
    // TODO: must check if it is a directory
    if (!Exists()) {
        Parent().MkDirs(mode);
        MkDir(mode);
    }
}

void TFsPath::ForceDelete() const {
    if (IsDirectory() && !IsSymlink()) {
        yvector<TFsPath> children;
        List(children);
        for (auto& i : children) {
            i.ForceDelete();
        }
    }
    DeleteIfExists();
}

void TFsPath::CopyTo(const TString& newPath, bool force) const {
    if (IsDirectory()) {
        if (force) {
            TFsPath(newPath).MkDirs();
        } else if (!TFsPath(newPath).IsDirectory()) {
            ythrow TIoException() << "Target path is not a directory " << newPath;
        }
        yvector<TFsPath> children;
        List(children);
        for (auto&& i : children) {
            i.CopyTo(newPath + "/" + i.GetName(), force);
        }
    } else {
        if (force) {
            TFsPath(newPath).Parent().MkDirs();
        } else {
            if (!TFsPath(newPath).Parent().IsDirectory()) {
                ythrow TIoException() << "Parent (" << TFsPath(newPath).Parent() << ") of a target path is not a directory " << newPath;
            }
            if (TFsPath(newPath).Exists()) {
                ythrow TIoException() << "Path already exists " << newPath;
            }
        }
        NFs::Copy(Path_, newPath);
    }
}

void TFsPath::ForceRenameTo(const TString& newPath) const {
    try {
        RenameTo(newPath);
    } catch (const TIoSystemError& /* error */) {
        CopyTo(newPath, true);
        ForceDelete();
    }
}

TFsPath TFsPath::Cwd() {
    return TFsPath(::NFs::CurrentWorkingDirectory());
}

const TPathSplit& TFsPath::PathSplit() const {
    return GetSplit();
}

template <>
void Out<TFsPath>(IOutputStream& os, const TFsPath& f) {
    os << f.GetPath();
}

template <>
TFsPath FromStringImpl<TFsPath>(const char* s, size_t len) {
    return TFsPath{TStringBuf{s, len}};
}

template <>
bool TryFromStringImpl(const char* s, size_t len, TFsPath& result) {
    try {
        result = TStringBuf{s, len};
        return true;
    } catch (std::exception&) {
        return false;
    }
}
