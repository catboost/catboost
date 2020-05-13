#include "fstat.h"
#include "file.h"
#include "sysstat.h"
#include "fs.h"

#include <library/cpp/unittest/registar.h>
#include <library/cpp/unittest/tests_data.h>

#include <util/folder/path.h>

Y_UNIT_TEST_SUITE(TestFileStat) {
    Y_UNIT_TEST(FileTest) {
        TString fileName = "f1.txt";
        TFileStat oFs;
        {
            TFile file(fileName.data(), OpenAlways | WrOnly);
            file.Write("1234567", 7);

            {
                TFileStat fs(file);
                UNIT_ASSERT(fs.IsFile());
                UNIT_ASSERT(!fs.IsDir());
                UNIT_ASSERT(!fs.IsSymlink());
                UNIT_ASSERT_VALUES_EQUAL(file.GetLength(), (i64)fs.Size);
                UNIT_ASSERT(fs.MTime >= fs.CTime);
                UNIT_ASSERT(fs.NLinks == 1);
                oFs = fs;
            }

            UNIT_ASSERT(file.IsOpen());
            UNIT_ASSERT_VALUES_EQUAL(file.GetLength(), 7);
            file.Close();
        }
        TFileStat cFs(fileName);
        UNIT_ASSERT(cFs.IsFile());
        UNIT_ASSERT(!cFs.IsDir());
        UNIT_ASSERT(!cFs.IsSymlink());
        UNIT_ASSERT_VALUES_EQUAL(cFs.Size, oFs.Size);
        UNIT_ASSERT(cFs.MTime >= oFs.MTime);
        UNIT_ASSERT_VALUES_EQUAL(cFs.CTime, oFs.CTime);
        UNIT_ASSERT_VALUES_EQUAL(cFs.NLinks, oFs.NLinks);
        UNIT_ASSERT_VALUES_EQUAL(cFs.Mode, oFs.Mode);
        UNIT_ASSERT_VALUES_EQUAL(cFs.Uid, oFs.Uid);
        UNIT_ASSERT_VALUES_EQUAL(cFs.Gid, oFs.Gid);
        UNIT_ASSERT_VALUES_EQUAL(cFs.INode, oFs.INode);
        UNIT_ASSERT(unlink(fileName.data()) == 0);
    }

    Y_UNIT_TEST(DirTest) {
        Mkdir("tmpd", MODE0777);
        TFileStat fs("tmpd");
        UNIT_ASSERT(!fs.IsFile());
        UNIT_ASSERT(fs.IsDir());
        UNIT_ASSERT(!fs.IsSymlink());
        //UNIT_ASSERT(fs.Size == 0); // it fails under unix
        UNIT_ASSERT(NFs::Remove("tmpd"));
        fs = TFileStat("tmpd");
        UNIT_ASSERT(!fs.IsFile());
        UNIT_ASSERT(!fs.IsDir());
        UNIT_ASSERT(!fs.IsSymlink());
        UNIT_ASSERT(fs.Size == 0);
        UNIT_ASSERT(fs.CTime == 0);
    }

    Y_UNIT_TEST(SymlinkToExistingFileTest) {
        const auto path = GetOutputPath() / "file_1";
        const auto link = GetOutputPath() / "symlink_1";
        TFile(path, EOpenModeFlag::CreateNew | EOpenModeFlag::RdWr);
        UNIT_ASSERT(NFs::SymLink(path, link));

        const TFileStat statNoFollow(link, false);
        UNIT_ASSERT_VALUES_EQUAL_C(false, statNoFollow.IsNull(), ToString(statNoFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(true, statNoFollow.IsFile(), ToString(statNoFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(false, statNoFollow.IsSymlink(), ToString(statNoFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(false, statNoFollow.IsDir(), ToString(statNoFollow.Mode));

        const TFileStat statFollow(link, true);
        UNIT_ASSERT_VALUES_EQUAL_C(false, statFollow.IsNull(), ToString(statFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(false, statFollow.IsFile(), ToString(statFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(true, statFollow.IsSymlink(), ToString(statFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(false, statFollow.IsDir(), ToString(statFollow.Mode));
    }

    Y_UNIT_TEST(SymlinkToNonExistingFileTest) {
        const auto path = GetOutputPath() / "file_2";
        const auto link = GetOutputPath() / "symlink_2";
        UNIT_ASSERT(NFs::SymLink(path, link));

        const TFileStat statNoFollow(link, false);
        UNIT_ASSERT_VALUES_EQUAL_C(true, statNoFollow.IsNull(), ToString(statNoFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(false, statNoFollow.IsFile(), ToString(statNoFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(false, statNoFollow.IsSymlink(), ToString(statNoFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(false, statNoFollow.IsDir(), ToString(statNoFollow.Mode));

        const TFileStat statFollow(link, true);
        UNIT_ASSERT_VALUES_EQUAL_C(false, statFollow.IsNull(), ToString(statFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(false, statFollow.IsFile(), ToString(statFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(true, statFollow.IsSymlink(), ToString(statFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(false, statFollow.IsDir(), ToString(statFollow.Mode));
    }

    Y_UNIT_TEST(SymlinkToFileThatCantExistTest) {
        const auto path = TFsPath("/path") / "that" / "does" / "not" / "exists";
        const auto link = GetOutputPath() / "symlink_3";
        UNIT_ASSERT(NFs::SymLink(path, link));

        const TFileStat statNoFollow(link, false);
        UNIT_ASSERT_VALUES_EQUAL_C(true, statNoFollow.IsNull(), ToString(statNoFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(false, statNoFollow.IsFile(), ToString(statNoFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(false, statNoFollow.IsSymlink(), ToString(statNoFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(false, statNoFollow.IsDir(), ToString(statNoFollow.Mode));

        const TFileStat statFollow(link, true);
        UNIT_ASSERT_VALUES_EQUAL_C(false, statFollow.IsNull(), ToString(statFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(false, statFollow.IsFile(), ToString(statFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(true, statFollow.IsSymlink(), ToString(statFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(false, statFollow.IsDir(), ToString(statFollow.Mode));
    }

    Y_UNIT_TEST(FileDoesNotExistTest) {
        const auto path = TFsPath("/path") / "that" / "does" / "not" / "exists";

        const TFileStat statNoFollow(path, false);
        UNIT_ASSERT_VALUES_EQUAL_C(true, statNoFollow.IsNull(), ToString(statNoFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(false, statNoFollow.IsFile(), ToString(statNoFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(false, statNoFollow.IsSymlink(), ToString(statNoFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(false, statNoFollow.IsDir(), ToString(statNoFollow.Mode));

        const TFileStat statFollow(path, true);
        UNIT_ASSERT_VALUES_EQUAL_C(true, statFollow.IsNull(), ToString(statFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(false, statFollow.IsFile(), ToString(statFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(false, statFollow.IsSymlink(), ToString(statFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(false, statFollow.IsDir(), ToString(statFollow.Mode));
    }

} // TestFileStat
