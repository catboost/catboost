#include "dirut.h"
#include "tempdir.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/string.h>
#include <util/memory/tempbuf.h>
#include <util/stream/file.h>

Y_UNIT_TEST_SUITE(TDirutTest) {
    Y_UNIT_TEST(TestRealPath) {
        UNIT_ASSERT(IsDir(RealPath(".")));
    }

    Y_UNIT_TEST(TestRealLocation) {
        UNIT_ASSERT(IsDir(RealLocation(".")));

        TTempDir tempDir;
        TString base = RealPath(tempDir());
        UNIT_ASSERT(!base.empty());

        if (base.back() == GetDirectorySeparator()) {
            base.pop_back();
        }

        TString path;
        TString pathNotNorm;

        path = base + GetDirectorySeparatorS() + "no_such_file";
        UNIT_ASSERT(NFs::Exists(GetDirName(path)));
        UNIT_ASSERT(!NFs::Exists(path));
        path = RealLocation(path);
        UNIT_ASSERT(NFs::Exists(GetDirName(path)));
        UNIT_ASSERT(!NFs::Exists(path));
        UNIT_ASSERT_EQUAL(GetDirName(path), base);

        pathNotNorm = base + GetDirectorySeparatorS() + "some_dir" + GetDirectorySeparatorS() + ".." + GetDirectorySeparatorS() + "no_such_file";
        MakeDirIfNotExist((base + GetDirectorySeparatorS() + "some_dir").data());
        pathNotNorm = RealLocation(pathNotNorm);
        UNIT_ASSERT(NFs::Exists(GetDirName(pathNotNorm)));
        UNIT_ASSERT(!NFs::Exists(pathNotNorm));
        UNIT_ASSERT_EQUAL(GetDirName(pathNotNorm), base);

        UNIT_ASSERT_EQUAL(path, pathNotNorm);

        path = base + GetDirectorySeparatorS() + "file";
        {
            TFixedBufferFileOutput file(path);
        }
        UNIT_ASSERT(NFs::Exists(GetDirName(path)));
        UNIT_ASSERT(NFs::Exists(path));
        UNIT_ASSERT(NFs::Exists(path));
        path = RealLocation(path);
        UNIT_ASSERT(NFs::Exists(GetDirName(path)));
        UNIT_ASSERT(NFs::Exists(path));
        UNIT_ASSERT_EQUAL(GetDirName(path), base);
    }

    void DoTest(const char* p, const char* base, const char* canon) {
        TString path(p);
        UNIT_ASSERT(resolvepath(path, base));
        UNIT_ASSERT(path == canon);
    }

    Y_UNIT_TEST(TestResolvePath) {
#ifdef _win_
        DoTest("bar", "c:\\foo\\baz", "c:\\foo\\baz\\bar");
        DoTest("c:\\foo\\bar", "c:\\bar\\baz", "c:\\foo\\bar");
#else
        DoTest("bar", "/foo/baz", "/foo/bar");
        DoTest("/foo/bar", "/bar/baz", "/foo/bar");

    #ifdef NDEBUG
        DoTest("bar", "./baz", "./bar");
        #if 0 // should we support, for consistency, single-label dirs
        DoTest("bar", "baz", "bar");
        #endif
    #endif
#endif
    }

    Y_UNIT_TEST(TestResolvePathRelative) {
        TTempDir tempDir;
        TTempBuf tempBuf;
        TString base = RealPath(tempDir());
        if (base.back() == GetDirectorySeparator()) {
            base.pop_back();
        }

        // File
        TString path = base + GetDirectorySeparatorS() + "file";
        {
            TFixedBufferFileOutput file(path);
        }
        ResolvePath("file", base.data(), tempBuf.Data(), false);
        UNIT_ASSERT_EQUAL(tempBuf.Data(), path);

        // Dir
        path = base + GetDirectorySeparatorS() + "dir";
        MakeDirIfNotExist(path.data());
        ResolvePath("dir", base.data(), tempBuf.Data(), true);
        UNIT_ASSERT_EQUAL(tempBuf.Data(), path + GetDirectorySeparatorS());

        // Absent file in existent dir
        path = base + GetDirectorySeparatorS() + "nofile";
        ResolvePath("nofile", base.data(), tempBuf.Data(), false);
        UNIT_ASSERT_EQUAL(tempBuf.Data(), path);
    }

    Y_UNIT_TEST(TestGetDirName) {
        UNIT_ASSERT_VALUES_EQUAL(".", GetDirName("parambambam"));
    }

    Y_UNIT_TEST(TestStripFileComponent) {
        static const TString tmpDir = "tmp_dir_for_tests";
        static const TString tmpSubDir = tmpDir + GetDirectorySeparatorS() + "subdir";
        static const TString tmpFile = tmpDir + GetDirectorySeparatorS() + "file";

        // creating tmp dir and subdirs
        MakeDirIfNotExist(tmpDir.data());
        MakeDirIfNotExist(tmpSubDir.data());
        {
            TFixedBufferFileOutput file(tmpFile);
        }

        UNIT_ASSERT_EQUAL(StripFileComponent(tmpDir), tmpDir + GetDirectorySeparatorS());
        UNIT_ASSERT_EQUAL(StripFileComponent(tmpSubDir), tmpSubDir + GetDirectorySeparatorS());
        UNIT_ASSERT_EQUAL(StripFileComponent(tmpFile), tmpDir + GetDirectorySeparatorS());

        RemoveDirWithContents(tmpDir);
    }
} // Y_UNIT_TEST_SUITE(TDirutTest)
