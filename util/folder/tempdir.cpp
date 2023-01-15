#include "tempdir.h"

#include "dirut.h"

#include <util/system/fs.h>
#include <util/system/maxlen.h>

TTempDir::TTempDir()
    : TTempDir(nullptr, TCreationToken{})
{
}

TTempDir::TTempDir(const char* prefix, TCreationToken)
    : TempDir()
    , Remove(true)
{
    char tempDir[MAX_PATH];
    if (MakeTempDir(tempDir, prefix) != 0) {
        ythrow TSystemError() << "Can't create temporary directory";
    }
    TempDir = tempDir;
}

TTempDir::TTempDir(const TString& tempDir)
    : TempDir(tempDir)
    , Remove(true)
{
    NFs::Remove(TempDir);
    MakeDirIfNotExist(TempDir.c_str());
}

TTempDir TTempDir::NewTempDir(const TString& root) {
    return {root.c_str(), TCreationToken{}};
}

void TTempDir::DoNotRemove() {
    Remove = false;
}

TTempDir::~TTempDir() {
    if (Remove) {
        RemoveDirWithContents(TempDir);
    }
}
