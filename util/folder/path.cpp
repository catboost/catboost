#include "dirut.h"
#include "path.h"
#include "pathsplit.h"

#include <util/generic/yexception.h>
#include <util/string/cast.h>
#include <util/string/escape.h>
#include <util/system/compiler.h>
#include <util/system/file.h>
#include <util/system/fs.h>

struct TFsPath::TSplit: public TAtomicRefCount<TSplit>, public TPathSplit {
    inline TSplit(const TStringBuf path)
        : TPathSplit(path)
    {
    }
    inline TSplit(const TString& path, const TSimpleIntrusivePtr<TSplit>& thatSplit, const TString::char_type* thatPathData) {
        for (const auto& thatPart : *thatSplit) {
            emplace_back(path.data() + (thatPart.data() - thatPathData), thatPart.size());
        }

        if (!thatSplit->Drive.empty()) {
            Drive = TStringBuf(path.data() + (thatSplit->Drive.data() - thatPathData), thatSplit->Drive.size());
        }

        IsAbsolute = thatSplit->IsAbsolute;
    }
};

void TFsPath::CheckDefined() const {
    if (!IsDefined()) {
        ythrow TIoException() << TStringBuf("must be defined");
    }
}

bool TFsPath::IsSubpathOf(const TFsPath& that) const {
    const TSplit& split = GetSplit();
    const TSplit& rsplit = that.GetSplit();

    if (rsplit.IsAbsolute != split.IsAbsolute) {
        return false;
    }

    if (rsplit.Drive != split.Drive) {
        return false;
    }

    if (rsplit.size() >= split.size()) {
        return false;
    }

    return std::equal(rsplit.begin(), rsplit.end(), split.begin());
}

bool TFsPath::IsNonStrictSubpathOf(const TFsPath& that) const {
    const TSplit& split = GetSplit();
    const TSplit& rsplit = that.GetSplit();

    if (rsplit.IsAbsolute != split.IsAbsolute) {
        return false;
    }

    if (rsplit.Drive != split.Drive) {
        return false;
    }

    if (rsplit.size() > split.size()) {
        return false;
    }

    return std::equal(rsplit.begin(), rsplit.end(), split.begin());
}

TFsPath TFsPath::RelativeTo(const TFsPath& root) const {
    TSplit split = GetSplit();
    const TSplit& rsplit = root.GetSplit();

    if (split.Reconstruct() == rsplit.Reconstruct()) {
        return TFsPath();
    }

    if (!this->IsSubpathOf(root)) {
        ythrow TIoException() << "path " << *this << " is not subpath of " << root;
    }

    split.erase(split.begin(), split.begin() + rsplit.size());
    split.IsAbsolute = false;

    return TFsPath(split.Reconstruct());
}

TFsPath TFsPath::RelativePath(const TFsPath& root) const {
    TSplit split = GetSplit();
    const TSplit& rsplit = root.GetSplit();
    size_t cnt = 0;

    while (split.size() > cnt && rsplit.size() > cnt && split[cnt] == rsplit[cnt]) {
        ++cnt;
    }
    bool absboth = split.IsAbsolute && rsplit.IsAbsolute;
    if (cnt == 0 && !absboth) {
        ythrow TIoException() << "No common parts in " << *this << " and " << root;
    }
    TString r;
    for (size_t i = 0; i < rsplit.size() - cnt; i++) {
        r += i == 0 ? ".." : "/..";
    }
    for (size_t i = cnt; i < split.size(); i++) {
        r += (i == 0 || i == cnt && rsplit.size() - cnt == 0 ? "" : "/");
        r += split[i];
    }
    return r.size() ? TFsPath(r) : TFsPath();
}

TFsPath TFsPath::Parent() const {
    if (!IsDefined()) {
        return TFsPath();
    }

    TSplit split = GetSplit();
    if (split.size()) {
        split.pop_back();
    }
    if (!split.size() && !split.IsAbsolute) {
        return TFsPath(".");
    }
    return TFsPath(split.Reconstruct());
}

TFsPath& TFsPath::operator/=(const TFsPath& that) {
    if (!IsDefined()) {
        *this = that;

    } else if (that.IsDefined() && that.GetPath() != ".") {
        if (!that.IsRelative()) {
            ythrow TIoException() << "path should be relative: " << that.GetPath();
        }

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
    if (!IsDefined()) {
        return TString();
    }

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
    if (!Split_) {
        InitSplit();
    }
    return *Split_;
}

static Y_FORCE_INLINE void VerifyPath(const TStringBuf path) {
    Y_ABORT_UNLESS(!path.Contains('\0'), "wrong format of TFsPath: %s", EscapeC(path).c_str());
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

TFsPath::TFsPath(const char* path)
    : Path_(path)
{
}

TFsPath::TFsPath(const TFsPath& that) {
    *this = that;
}

TFsPath::TFsPath(TFsPath&& that) {
    *this = std::move(that);
}

TFsPath& TFsPath::operator=(const TFsPath& that) {
    Path_ = that.Path_;
    if (that.Split_) {
        Split_ = new TSplit(Path_, that.Split_, that.Path_.begin());
    } else {
        Split_ = nullptr;
    }
    return *this;
}

TFsPath& TFsPath::operator=(TFsPath&& that) {
#ifdef TSTRING_IS_STD_STRING
    const auto thatPathData = that.Path_.data();
    Path_ = std::move(that.Path_);
    if (that.Split_) {
        if (Path_.data() == thatPathData) { // Path_ moved,  can move Split_
            Split_ = std::move(that.Split_);
        } else { // Path_ copied, rebuild Split_ using that.Split_
            Split_ = new TSplit(Path_, that.Split_, that.Path_.data());
        }
    } else {
        Split_ = nullptr;
    }
#else
    Path_ = std::move(that.Path_);
    Split_ = std::move(that.Split_);
#endif
    return *this;
}

TFsPath TFsPath::Child(const TString& name) const {
    if (!name) {
        ythrow TIoException() << "child name must not be empty";
    }

    return *this / name;
}

struct TClosedir {
    static void Destroy(DIR* dir) {
        if (dir) {
            if (0 != closedir(dir)) {
                ythrow TIoSystemError() << "failed to closedir";
            }
        }
    }
};

void TFsPath::ListNames(TVector<TString>& children) const {
    CheckDefined();
    THolder<DIR, TClosedir> dir(opendir(this->c_str()));
    if (!dir) {
        ythrow TIoSystemError() << "failed to opendir " << Path_;
    }

    for (;;) {
        struct dirent de;
        struct dirent* ok;
        // TODO(yazevnul|IGNIETFERRO-1070): remove these macroses by replacing `readdir_r` with proper
        // alternative
        Y_PRAGMA_DIAGNOSTIC_PUSH
        Y_PRAGMA_NO_DEPRECATED
        int r = readdir_r(dir.Get(), &de, &ok);
        Y_PRAGMA_DIAGNOSTIC_POP
        if (r != 0) {
            ythrow TIoSystemError() << "failed to readdir " << Path_;
        }
        if (ok == nullptr) {
            return;
        }
        TString name(de.d_name);
        if (name == "." || name == "..") {
            continue;
        }
        children.push_back(name);
    }
}

bool TFsPath::Contains(const TString& component) const {
    if (!IsDefined()) {
        return false;
    }

    TFsPath path = *this;
    while (path.Parent() != path) {
        if (path.GetName() == component) {
            return true;
        }

        path = path.Parent();
    }

    return false;
}

void TFsPath::List(TVector<TFsPath>& files) const {
    TVector<TString> names;
    ListNames(names);
    for (auto& name : names) {
        files.push_back(Child(name));
    }
}

void TFsPath::RenameTo(const TString& newPath) const {
    CheckDefined();
    if (!newPath) {
        ythrow TIoException() << "bad new file name";
    }
    if (!NFs::Rename(Path_, newPath)) {
        ythrow TIoSystemError() << "failed to rename " << Path_ << " to " << newPath;
    }
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
    return ::RealPath(*this);
}

TFsPath TFsPath::RealLocation() const {
    CheckDefined();
    return ::RealLocation(*this);
}

TFsPath TFsPath::ReadLink() const {
    CheckDefined();

    if (!IsSymlink()) {
        ythrow TIoException() << "not a symlink " << *this;
    }

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
    return IsDefined() && TFileStat(GetPath().data()).IsDir();
}

bool TFsPath::IsFile() const {
    return IsDefined() && TFileStat(GetPath().data()).IsFile();
}

bool TFsPath::IsSymlink() const {
    return IsDefined() && TFileStat(GetPath().data(), true).IsSymlink();
}

void TFsPath::DeleteIfExists() const {
    if (!IsDefined()) {
        return;
    }

    ::unlink(this->c_str());
    ::rmdir(this->c_str());
    if (Exists()) {
        ythrow TIoException() << "failed to delete " << Path_;
    }
}

void TFsPath::MkDir(const int mode) const {
    CheckDefined();
    if (!Exists()) {
        int r = Mkdir(this->c_str(), mode);
        if (r != 0) {
            // TODO (stanly) will still fail on Windows because
            // LastSystemError() returns windows specific ERROR_ALREADY_EXISTS
            // instead of EEXIST.
            if (LastSystemError() != EEXIST) {
                ythrow TIoSystemError() << "could not create directory " << Path_;
            }
        }
    }
}

void TFsPath::MkDirs(const int mode) const {
    CheckDefined();
    if (!Exists()) {
        Parent().MkDirs(mode);
        MkDir(mode);
    }
}

void TFsPath::ForceDelete() const {
    if (!IsDefined()) {
        return;
    }

    TFileStat stat(GetPath().c_str(), true);
    if (stat.IsNull()) {
        const int err = LastSystemError();
#ifdef _win_
        if (err == ERROR_FILE_NOT_FOUND || err == ERROR_PATH_NOT_FOUND) {
#else
        if (err == ENOENT) {
#endif
            return;
        } else {
            ythrow TIoException() << "failed to stat " << Path_;
        }
    }

    bool succ;
    if (stat.IsDir()) {
        TVector<TFsPath> children;
        List(children);
        for (auto& i : children) {
            i.ForceDelete();
        }
        succ = ::rmdir(this->c_str()) == 0;
    } else {
        succ = ::unlink(this->c_str()) == 0;
    }

    if (!succ && LastSystemError()) {
        ythrow TIoException() << "failed to delete " << Path_;
    }
}

void TFsPath::CopyTo(const TString& newPath, bool force) const {
    if (IsDirectory()) {
        if (force) {
            TFsPath(newPath).MkDirs();
        } else if (!TFsPath(newPath).IsDirectory()) {
            ythrow TIoException() << "Target path is not a directory " << newPath;
        }
        TVector<TFsPath> children;
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
