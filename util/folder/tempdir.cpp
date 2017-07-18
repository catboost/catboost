#include "tempdir.h"

#include "dirut.h"

#include <util/system/fs.h>
#include <util/system/maxlen.h>

TTempDir::TTempDir()
    : TempDir()
    , Remove(true)
{
    char tempDir[MAX_PATH];
    if (MakeTempDir(tempDir, nullptr) != 0)
        ythrow TSystemError() << "Can't create temporary directory";
    TempDir = tempDir;
}

TTempDir::TTempDir(const TString& tempDir)
    : TempDir(tempDir)
    , Remove(true)
{
    NFs::Remove(TempDir);
    MakeDirIfNotExist(~TempDir);
}

void TTempDir::DoNotRemove() {
    Remove = false;
}

TTempDir::~TTempDir() {
    if (Remove) {
        RemoveDirWithContents(TempDir);
    }
}
