#include <library/cpp/testing/common/env.h>
#include <library/cpp/testing/common/scope.h>

#include <util/folder/dirut.h>
#include <util/system/env.h>
#include <util/system/execpath.h>

#include <library/cpp/testing/gtest/gtest.h>

TEST(Runtime, ArcadiaSourceRoot) {
    NTesting::TScopedEnvironment contextGuard("YA_TEST_CONTEXT_FILE", ""); // remove context filename
    {
        auto tmpDir = ::GetSystemTempDir();
        NTesting::TScopedEnvironment guard("ARCADIA_SOURCE_ROOT", tmpDir);
        Singleton<NPrivate::TTestEnv>()->ReInitialize();
        EXPECT_EQ(tmpDir, ArcadiaSourceRoot());
    }
    {
        NTesting::TScopedEnvironment guard("ARCADIA_SOURCE_ROOT", "");
        Singleton<NPrivate::TTestEnv>()->ReInitialize();
        EXPECT_FALSE(ArcadiaSourceRoot().empty());
    }
}

TEST(Runtime, BuildRoot) {
    NTesting::TScopedEnvironment contextGuard("YA_TEST_CONTEXT_FILE", ""); // remove context filename
    {
        auto tmpDir = ::GetSystemTempDir();
        NTesting::TScopedEnvironment guard("ARCADIA_BUILD_ROOT", tmpDir);
        Singleton<NPrivate::TTestEnv>()->ReInitialize();
        EXPECT_EQ(tmpDir, BuildRoot());
    }
    {
        NTesting::TScopedEnvironment guard("ARCADIA_BUILD_ROOT", "");
        Singleton<NPrivate::TTestEnv>()->ReInitialize();
        EXPECT_FALSE(BuildRoot().empty());
    }
}

TEST(Runtime, BinaryPath) {
    NTesting::TScopedEnvironment contextGuard("YA_TEST_CONTEXT_FILE", ""); // remove context filename
    Singleton<NPrivate::TTestEnv>()->ReInitialize();
    EXPECT_TRUE(TFsPath(BinaryPath("library/cpp/testing/common/ut")).Exists());
}

TEST(Runtime, GetArcadiaTestsData) {
    NTesting::TScopedEnvironment contextGuard("YA_TEST_CONTEXT_FILE", ""); // remove context filename
    {
        auto tmpDir = ::GetSystemTempDir();
        NTesting::TScopedEnvironment guard("ARCADIA_TESTS_DATA_DIR", tmpDir);
        Singleton<NPrivate::TTestEnv>()->ReInitialize();
        EXPECT_EQ(tmpDir, GetArcadiaTestsData());
    }
    {
        NTesting::TScopedEnvironment guard("ARCADIA_TESTS_DATA_DIR", "");
        Singleton<NPrivate::TTestEnv>()->ReInitialize();
        auto path = GetArcadiaTestsData();
        // it is not error if path is empty
        const bool ok = (path.empty() || GetBaseName(path) == "arcadia_tests_data");
        EXPECT_TRUE(ok);
    }
}

TEST(Runtime, GetWorkPath) {
    NTesting::TScopedEnvironment contextGuard("YA_TEST_CONTEXT_FILE", ""); // remove context filename
    {
        auto tmpDir = ::GetSystemTempDir();
        NTesting::TScopedEnvironment guard("TEST_WORK_PATH", tmpDir);
        Singleton<NPrivate::TTestEnv>()->ReInitialize();
        EXPECT_EQ(tmpDir, GetWorkPath());
    }
    {
        NTesting::TScopedEnvironment guard("TEST_WORK_PATH", "");
        Singleton<NPrivate::TTestEnv>()->ReInitialize();
        EXPECT_TRUE(!GetWorkPath().empty());
    }
}

TEST(Runtime, GetOutputPath) {
    NTesting::TScopedEnvironment contextGuard("YA_TEST_CONTEXT_FILE", ""); // remove context filename
    Singleton<NPrivate::TTestEnv>()->ReInitialize();
    EXPECT_EQ(GetOutputPath().Basename(), "testing_out_stuff");
}

TEST(Runtime, GetRamDrivePath) {
    NTesting::TScopedEnvironment contextGuard("YA_TEST_CONTEXT_FILE", ""); // remove context filename
    auto tmpDir = ::GetSystemTempDir();
    NTesting::TScopedEnvironment guard("YA_TEST_RAM_DRIVE_PATH", tmpDir);
    Singleton<NPrivate::TTestEnv>()->ReInitialize();
    EXPECT_EQ(tmpDir, GetRamDrivePath());
}

TEST(Runtime, GetOutputRamDrivePath) {
    NTesting::TScopedEnvironment contextGuard("YA_TEST_CONTEXT_FILE", ""); // remove context filename
    auto tmpDir = ::GetSystemTempDir();
    NTesting::TScopedEnvironment guard("YA_TEST_OUTPUT_RAM_DRIVE_PATH", tmpDir);
    Singleton<NPrivate::TTestEnv>()->ReInitialize();
    EXPECT_EQ(tmpDir, GetOutputRamDrivePath());
}
