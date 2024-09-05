#pragma once

#include <util/generic/flags.h>
#include <util/generic/string.h>
#include <util/generic/yexception.h>

namespace NFs {
    enum EFilePermission {
        FP_ALL_EXEC = 01,
        FP_ALL_WRITE = 02,
        FP_ALL_READ = 04,
        FP_GROUP_READ = 040,
        FP_GROUP_WRITE = 020,
        FP_GROUP_EXEC = 010,
        FP_OWNER_READ = 0400,
        FP_OWNER_WRITE = 0200,
        FP_OWNER_EXEC = 0100,

        FP_COMMON_FILE = 0777,
        FP_SECRET_FILE = 0700,
        FP_NONSECRET_FILE = 0744,
    };

    Y_DECLARE_FLAGS(EFilePermissions, EFilePermission);

    /// Add executable bit
    ///
    /// @param[in] path  Path to mark as executable
    /// @param[in] exec  New value of execution bit
    /// @returns         true if bit has changed or false otherwise
    bool SetExecutable(const TString& path, bool exec);

    /// Remove a file or empty directory
    ///
    /// @param[in] path  Path to file or directory
    /// @returns         true on success or false otherwise
    /// LastSystemError() is set in case of failure
    bool Remove(const TString& path);

    /// Remove a file or directory with contents
    /// Does nothing if path does not exist
    ///
    /// @param[in] path  Path to file or directory
    /// @throws
    void RemoveRecursive(const TString& path);

    /// Creates directory
    ///
    /// @param[in] path  Path to the directory
    /// @param[in] mode  Access permissions field; NOTE: ignored on win
    /// @returns         true on success or false otherwise
    bool MakeDirectory(const TString& path, EFilePermissions mode);

    /// Creates directory
    ///
    /// @param[in] path  Path to the directory
    /// @returns         true on success or false otherwise
    /// NOTE: access permissions is set to 0777
    inline bool MakeDirectory(const TString& path) {
        return MakeDirectory(path, FP_COMMON_FILE);
    }

    /// Creates directory and all non-existings parents
    ///
    /// @param[in] path          Path to be created
    /// @param[in] mode          Access permissions field; NOTE: ignored on win
    /// @param[in] alwaysCreate  Throw if path already exists or failed to create
    /// @returns                 true if target path created or exists (and directory)
    bool MakeDirectoryRecursive(const TString& path, EFilePermissions mode, bool alwaysCreate);

    /// Creates directory and all non-existings parents
    ///
    /// @param[in] path          Path to be created
    /// @param[in] mode          Access permissions field; NOTE: ignored on win
    /// @returns                 true if target path created or exists (and directory)
    inline bool MakeDirectoryRecursive(const TString& path, EFilePermissions mode) {
        return MakeDirectoryRecursive(path, mode, false);
    }

    /// Creates directory and all non-existings parents
    ///
    /// @param[in] path          Path to be created
    /// @returns                 true if target path created or exists (and directory)
    inline bool MakeDirectoryRecursive(const TString& path) {
        return MakeDirectoryRecursive(path, FP_COMMON_FILE, false);
    }

    /// Rename a file or directory.
    /// Removes newPath if it exists
    ///
    /// @param[in] oldPath  Path to file or directory to rename
    /// @param[in] newPath  New path of file or directory
    /// @returns            true on success or false otherwise
    /// LastSystemError() is set in case of failure
    bool Rename(const TString& oldPath, const TString& newPath);

    /// Creates a new directory entry for a file
    /// or creates a new one with the same content
    ///
    /// @param[in] existingPath  Path to an existing file
    /// @param[in] newPath       New path of file
    void HardLinkOrCopy(const TString& existingPath, const TString& newPath);

    /// Creates a new directory entry for a file
    ///
    /// @param[in] existingPath  Path to an existing file
    /// @param[in] newPath       New path of file
    /// @returns                 true if new link was created or false otherwise
    /// LastSystemError() is set in case of failure
    bool HardLink(const TString& existingPath, const TString& newPath);

    /// Creates a symlink to a file
    ///
    /// @param[in] targetPath    Path to a target file
    /// @param[in] linkPath      Path of symlink
    /// @returns                 true if new link was created or false otherwise
    /// LastSystemError() is set in case of failure
    bool SymLink(const TString& targetPath, const TString& linkPath);

    /// Reads value of a symbolic link
    ///
    /// @param[in] path    Path to a symlink
    /// @returns           File path that a symlink points to
    TString ReadLink(const TString& path);

    /// Append contents of a file to a new file
    ///
    /// @param[in] dstPath  Path to a destination file
    /// @param[in] srcPath  Path to a source file
    void Cat(const TString& dstPath, const TString& srcPath);

    /// Copy contents of a file to a new file
    ///
    /// @param[in] existingPath  Path to an existing file
    /// @param[in] newPath       New path of file
    void Copy(const TString& existingPath, const TString& newPath);

    /// Returns path to the current working directory
    ///
    /// Note: is not threadsafe
    TString CurrentWorkingDirectory();

    /// Changes current working directory
    ///
    /// @param[in] path          Path for new cwd
    /// Note: is not threadsafe
    void SetCurrentWorkingDirectory(const TString& path);

    /// Checks if file exists
    ///
    /// @param[in] path          Path to check
    bool Exists(const TString& path);

    /// Ensures that file exists
    ///
    /// @param[in] path          Path to check
    /// @returns                 input argument
    inline const TString& EnsureExists(const TString& path) {
        Y_ENSURE_EX(Exists(path), TFileError{} << "Path " << path << " does not exists (checked from cwd:" << NFs::CurrentWorkingDirectory() << ")");
        return path;
    }
} // namespace NFs

Y_DECLARE_OPERATORS_FOR_FLAGS(NFs::EFilePermissions);
