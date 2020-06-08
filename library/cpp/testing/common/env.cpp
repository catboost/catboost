#include "env.h"

#include <util/folder/dirut.h>
#include <util/system/env.h>
#include <util/system/fs.h>
#include <util/folder/path.h>
#include <util/generic/singleton.h>


namespace {
    class TTestEnv {
    public:
        TTestEnv() {
            const TString fromEnv = GetEnv("YA_TEST_RUNNER");
            IsRunningFromTest = (fromEnv == "1");
        }

        bool IsRunningFromTest;
    };

    TString GetCwd() {
        try {
            return NFs::CurrentWorkingDirectory();
        } catch (...) {
            return {};
        }
    }
} //anonymous namespace

TString ArcadiaFromCurrentLocation(TStringBuf where, TStringBuf path) {
    return (TFsPath(ArcadiaSourceRoot()) / TFsPath(where).Parent() / path).Fix();
}

TString BinaryPath(TStringBuf path) {
    return (TFsPath(BuildRoot()) / path).Fix();
}

TString GetArcadiaTestsData() {
    TString envPath = GetEnv("ARCADIA_TESTS_DATA_DIR");
    if (envPath) {
        return envPath;
    }

    TString path = GetCwd();
    const char pathsep = GetDirectorySeparator();
    while (!path.empty()) {
        TString dataDir = path + "/arcadia_tests_data";
        if (IsDir(dataDir)) {
            return dataDir;
        }

        size_t pos = path.find_last_of(pathsep);
        if (pos == TString::npos) {
            pos = 0;
        }
        path.erase(pos);
    }

    return {};
}

TString GetWorkPath() {
    TString envPath = GetEnv("TEST_WORK_PATH");
    if (envPath) {
        return envPath;
    }
    return GetCwd();
}

TFsPath GetYaPath() {
    TString envPath = GetEnv("YA_CACHE_DIR");
    if (!envPath) {
        envPath = GetHomeDir() + "/.ya";
    }
    return envPath;
}

TFsPath GetOutputPath() {
    return GetWorkPath() + "/testing_out_stuff";
}

TString GetRamDrivePath() {
    return GetEnv("YA_TEST_RAM_DRIVE_PATH");
}

TString GetOutputRamDrivePath() {
    return GetEnv("YA_TEST_OUTPUT_RAM_DRIVE_PATH");
}

bool FromYaTest() {
    return Singleton<TTestEnv>()->IsRunningFromTest;
}
