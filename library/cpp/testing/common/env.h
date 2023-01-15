#pragma once

#include <unordered_map>

#include <util/folder/path.h>
#include <util/generic/string.h>
#include <util/generic/strbuf.h>
#include <util/system/src_location.h>

// @brief return full path to arcadia root
TString ArcadiaSourceRoot();

// @brief return full path for file or folder specified by known source location `where` and `path` which is relative to parent folder of `where`
//        for the instance: there is 2 files in folder test example_ut.cpp and example.data, so full path to test/example.data can be obtained
//        from example_ut.cpp as ArcadiaFromCurrentLocation(__SOURCE_FILE__, "example.data")
TString ArcadiaFromCurrentLocation(TStringBuf where, TStringBuf path);

// @brief return build folder path
TString BuildRoot();

// @brief return full path to built artefact, where path is relative from arcadia root
TString BinaryPath(TStringBuf path);

// @brief return true if environment is testenv otherwise false
bool FromYaTest();

// @brief returns TestsData dir (from env:ARCADIA_TESTS_DATA_DIR or path to existing folder `arcadia_tests_data` within parent folders)
TString GetArcadiaTestsData();

// @brief return current working dir (from env:TEST_WORK_PATH or cwd)
TString GetWorkPath();

// @brief return tests output path (workdir + testing_out_stuff)
TFsPath GetOutputPath();

// @brief return path from env:YA_TEST_RAM_DRIVE_PATH
const TString& GetRamDrivePath();

// @brief return path from env:YA_TEST_OUTPUT_RAM_DRIVE_PATH
const TString& GetOutputRamDrivePath();

// @brief return test parameter by name. If not exists, return an empty string
TString GetTestParam(TStringBuf name);

// @brief return path to the gdb
const TString& GdbPath();

#define SRC_(path) ArcadiaFromCurrentLocation(__SOURCE_FILE__, path)

namespace NPrivate {
    class TTestEnv {
    public:
        TTestEnv();

        void ReInitialize();

        bool IsRunningFromTest;
        TString ArcadiaTestsDataDir;
        TString SourceRoot;
        TString BuildRoot;
        TString WorkPath;
        TString RamDrivePath;
        TString TestOutputRamDrivePath;
        TString GdbPath;
        std::unordered_map<TString, TString> TestParameters;
    };

    TString GetCwd();

    const TTestEnv& GetTestEnv();
}
