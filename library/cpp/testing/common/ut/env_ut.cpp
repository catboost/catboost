#include <library/cpp/testing/common/env.h>
#include <library/cpp/testing/common/scope.h>

#include <util/folder/dirut.h>
#include <util/system/env.h>
#include <util/system/execpath.h>

#include <library/cpp/testing/gtest/gtest.h>

TEST(Runtime, ArcadiaSourceRoot) {
    {
        auto tmpDir = ::GetSystemTempDir();
        NTesting::TScopedEnvironment guard("ARCADIA_SOURCE_ROOT", tmpDir);
        EXPECT_EQ(tmpDir, ArcadiaSourceRoot());
    }
    {
        NTesting::TScopedEnvironment guard("ARCADIA_SOURCE_ROOT", "");
        EXPECT_FALSE(ArcadiaSourceRoot().empty());
    }
}

TEST(Runtime, BuildRoot) {
    {
        auto tmpDir = ::GetSystemTempDir();
        NTesting::TScopedEnvironment guard("ARCADIA_BUILD_ROOT", tmpDir);
        EXPECT_EQ(tmpDir, BuildRoot());
    }
    {
        NTesting::TScopedEnvironment guard("ARCADIA_BUILD_ROOT", "");
        EXPECT_FALSE(BuildRoot().empty());
    }
}

TEST(Runtime, BinaryPath) {
    EXPECT_TRUE(TFsPath(BinaryPath("library/cpp/testing/common/ut")).Exists());
}

TEST(Runtime, GetArcadiaTestsData) {
    {
        auto tmpDir = ::GetSystemTempDir();
        NTesting::TScopedEnvironment guard("ARCADIA_TESTS_DATA_DIR", tmpDir);
        EXPECT_EQ(tmpDir, GetArcadiaTestsData());
    }
    {
        NTesting::TScopedEnvironment guard("ARCADIA_TESTS_DATA_DIR", "");
        auto path = GetArcadiaTestsData();
        // it is not error if path is empty
        const bool ok = (path.empty() || GetBaseName(path) == "arcadia_tests_data");
        EXPECT_TRUE(ok);
    }
}

TEST(Runtime, GetWorkPath) {
    {
        auto tmpDir = ::GetSystemTempDir();
        NTesting::TScopedEnvironment guard("TEST_WORK_PATH", tmpDir);
        EXPECT_EQ(tmpDir, GetWorkPath());
    }
    {
        NTesting::TScopedEnvironment guard("TEST_WORK_PATH", "");
        EXPECT_TRUE(!GetWorkPath().empty());
    }
}

TEST(Runtime, GetYaPath) {
    {
        auto tmpDir = ::GetSystemTempDir();
        NTesting::TScopedEnvironment guard("YA_CACHE_DIR", tmpDir);
        EXPECT_EQ(tmpDir, GetYaPath().GetPath());
    }
    {
        NTesting::TScopedEnvironment guard("YA_CACHE_DIR", "");
        EXPECT_EQ(GetYaPath().Basename(), ".ya");
    }
}

TEST(Runtime, GetOutputPath) {
    EXPECT_EQ(GetOutputPath().Basename(), "testing_out_stuff");
}

TEST(Runtime, GetRamDrivePath) {
    auto tmpDir = ::GetSystemTempDir();
    NTesting::TScopedEnvironment guard("YA_TEST_RAM_DRIVE_PATH", tmpDir);
    EXPECT_EQ(tmpDir, GetRamDrivePath());
}

TEST(Runtime, GetOutputRamDrivePath) {
    auto tmpDir = ::GetSystemTempDir();
    NTesting::TScopedEnvironment guard("YA_TEST_OUTPUT_RAM_DRIVE_PATH", tmpDir);
    EXPECT_EQ(tmpDir, GetOutputRamDrivePath());
}
