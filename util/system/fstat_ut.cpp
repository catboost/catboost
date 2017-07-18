#include "fstat.h"
#include "file.h"
#include "sysstat.h"
#include "fs.h"

#include <library/unittest/registar.h>

SIMPLE_UNIT_TEST_SUITE(TestFileStat) {
    SIMPLE_UNIT_TEST(FileTest) {
        TString fileName = "f1.txt";
        TFileStat oFs;
        {
            TFile file(~fileName, OpenAlways | WrOnly);
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
        UNIT_ASSERT(unlink(~fileName) == 0);
    }

    SIMPLE_UNIT_TEST(DirTest) {
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

} // TestFileStat
