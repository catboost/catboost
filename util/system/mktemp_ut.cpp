#include <library/cpp/testing/unittest/registar.h>

#include <filesystem>

#include "tempfile.h"

#include <util/folder/dirut.h>

Y_UNIT_TEST_SUITE(MakeTempFileSuite) {
    static const char TestDir[] = "Test";
    static const char Prefix[] = "PREFIX_____PREFIX";
    static const char Extension[] = "txt";
    static const unsigned int PathSegmentSizeNormal = 55;
    static const unsigned int PathSegmentSizeLong = 255;

    Y_UNIT_TEST(TestMakeTempName) {
        const TFsPath systemTemp{GetSystemTempDir()};
        UNIT_ASSERT(systemTemp.Exists());

        for (auto dirNameLength : {PathSegmentSizeNormal, PathSegmentSizeLong}) {
            const TFsPath testDir{systemTemp / TestDir};
            testDir.MkDir();
            UNIT_ASSERT(testDir.Exists());
            Y_DEFER {
                std::filesystem::remove_all(testDir.c_str());
            };

            const TString dirName(dirNameLength, 'X');
            const TFsPath dirPath = testDir / dirName;
            UNIT_ASSERT(std::filesystem::create_directory(dirPath.GetPath().c_str()));

            TString tempFilePath;
            try {
                tempFilePath = MakeTempName(dirPath.c_str(), Prefix, Extension);
            } catch (const TSystemError& ex) {
                Cerr << "Unexpected exception: " << ex.what() << Endl;
            }
            UNIT_ASSERT(TFsPath{tempFilePath}.Exists());
        }
    }
} // Y_UNIT_TEST_SUITE(MakeTempFileSuite)
