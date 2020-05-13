#include "fts.h"
#include "dirut.h"
#include "tempdir.h"

#include <library/cpp/unittest/registar.h>

#include <util/system/file.h>
#include <util/system/mktemp.h>
#include <util/generic/string.h>

class TFtsTest: public TTestBase {
    UNIT_TEST_SUITE(TFtsTest);
    UNIT_TEST(TestSimple);
    UNIT_TEST_SUITE_END();

public:
    void TestSimple();
};

void MakeFile(const char* path) {
    TFile(path, CreateAlways);
}

//There potentially could be problems in listing order on different platforms
int FtsCmp(const FTSENT** ent1, const FTSENT** ent2) {
    return strcmp((*ent1)->fts_accpath, (*ent2)->fts_accpath);
}

void CheckEnt(FTSENT* ent, const char* name, int type) {
    UNIT_ASSERT(ent);
    UNIT_ASSERT_STRINGS_EQUAL(ent->fts_path, name);
    UNIT_ASSERT_EQUAL(ent->fts_info, type);
}

class TFileTree {
public:
    TFileTree(char* const* argv, int options, int (*compar)(const FTSENT**, const FTSENT**)) {
        Fts_ = yfts_open(argv, options, compar);
    }

    ~TFileTree() {
        yfts_close(Fts_);
    }

    FTS* operator()() {
        return Fts_;
    }

private:
    FTS* Fts_;
};

void TFtsTest::TestSimple() {
    const char* dotPath[2] = {"." LOCSLASH_S, nullptr};
    TFileTree currentDirTree((char* const*)dotPath, 0, FtsCmp);
    UNIT_ASSERT(currentDirTree());
    TTempDir tempDir = MakeTempName(yfts_read(currentDirTree())->fts_path);
    MakeDirIfNotExist(tempDir().data());
    MakeDirIfNotExist((tempDir() + LOCSLASH_S "dir1").data());
    MakeFile((tempDir() + LOCSLASH_S "dir1" LOCSLASH_S "file1").data());
    MakeFile((tempDir() + LOCSLASH_S "dir1" LOCSLASH_S "file2").data());
    MakeDirIfNotExist((tempDir() + LOCSLASH_S "dir2").data());
    MakeFile((tempDir() + LOCSLASH_S "dir2" LOCSLASH_S "file3").data());
    MakeFile((tempDir() + LOCSLASH_S "dir2" LOCSLASH_S "file4").data());

    const char* path[2] = {tempDir().data(), nullptr};
    TFileTree fileTree((char* const*)path, 0, FtsCmp);
    UNIT_ASSERT(fileTree());
    CheckEnt(yfts_read(fileTree()), tempDir().data(), FTS_D);
    CheckEnt(yfts_read(fileTree()), (tempDir() + LOCSLASH_S "dir1").data(), FTS_D);
    CheckEnt(yfts_read(fileTree()), (tempDir() + LOCSLASH_S "dir1" LOCSLASH_S "file1").data(), FTS_F);
    CheckEnt(yfts_read(fileTree()), (tempDir() + LOCSLASH_S "dir1" LOCSLASH_S "file2").data(), FTS_F);
    CheckEnt(yfts_read(fileTree()), (tempDir() + LOCSLASH_S "dir1").data(), FTS_DP);
    CheckEnt(yfts_read(fileTree()), (tempDir() + LOCSLASH_S "dir2").data(), FTS_D);
    CheckEnt(yfts_read(fileTree()), (tempDir() + LOCSLASH_S "dir2" LOCSLASH_S "file3").data(), FTS_F);
    CheckEnt(yfts_read(fileTree()), (tempDir() + LOCSLASH_S "dir2" LOCSLASH_S "file4").data(), FTS_F);
    CheckEnt(yfts_read(fileTree()), (tempDir() + LOCSLASH_S "dir2").data(), FTS_DP);
    CheckEnt(yfts_read(fileTree()), (tempDir()).data(), FTS_DP);
    UNIT_ASSERT_EQUAL(yfts_read(fileTree()), nullptr);
}

UNIT_TEST_SUITE_REGISTRATION(TFtsTest);
