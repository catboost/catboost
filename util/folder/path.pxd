from util.generic.string cimport TString, TStringBuf
from util.generic.vector cimport TVector


# NOTE (danila-eremin) Currently not possible to use `const` and `except +` at the same time, so some function not marked const
cdef extern from "util/folder/path.h" nogil:
    cdef cppclass TFsPath:
        TFsPath() except +
        TFsPath(const TString&) except +
        TFsPath(const TStringBuf) except +
        TFsPath(const char*) except +

        void CheckDefined() except +

        bint IsDefined() const
        bint operator bool() const

        const char* c_str() const

        bint operator==(const TFsPath&) const
        bint operator!=(const TFsPath&) const

        # NOTE (danila-eremin) operator `/=` Not supported
        # TFsPath& operator/=(const TFsPath&) const

        TFsPath operator/(const TFsPath&, const TFsPath&) except +

        # NOTE (danila-eremin) TPathSplit needed
        # const TPathSplit& PathSplit() const

        TFsPath& Fix() except +

        const TString& GetPath() const
        TString GetName() const

        TString GetExtension() const

        bint IsAbsolute() const
        bint IsRelative() const

        bint IsSubpathOf(const TFsPath&) const
        bint IsNonStrictSubpathOf(const TFsPath&) const
        bint IsContainerOf(const TFsPath&) const

        TFsPath RelativeTo(const TFsPath&) except +
        TFsPath RelativePath(const TFsPath&) except +

        TFsPath Parent() const

        TString Basename() const
        TString Dirname() const

        TFsPath Child(const TString&) except +

        void MkDir() except +
        void MkDir(const int) except +
        void MkDirs() except +
        void MkDirs(const int) except +

        void List(TVector[TFsPath]&) except +
        void ListNames(TVector[TString]&) except +

        bint Contains(const TString&) const

        void DeleteIfExists() except +
        void ForceDelete() except +

        # NOTE (danila-eremin) TFileStat needed
        # bint Stat(TFileStat&) const

        bint Exists() const
        bint IsDirectory() const
        bint IsFile() const
        bint IsSymlink() const
        void CheckExists() except +

        void RenameTo(const TString&) except +
        void RenameTo(const char*) except +
        void RenameTo(const TFsPath&) except +
        void ForceRenameTo(const TString&) except +

        void CopyTo(const TString&, bint) except +

        void Touch() except +

        TFsPath RealPath() except +
        TFsPath RealLocation() except +
        TFsPath ReadLink() except +

        @staticmethod
        TFsPath Cwd() except +

        void Swap(TFsPath&)
