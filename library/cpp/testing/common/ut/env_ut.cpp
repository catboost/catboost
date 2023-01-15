#include <library/cpp/testing/common/env.h>

#include <util/folder/dirut.h>
#include <util/system/env.h>
#include <util/system/execpath.h>

#include <library/cpp/testing/gtest/gtest.h>

namespace {
    struct TEnvGuard {
        TEnvGuard(const TString& name, const TString& value)
            : Name_(name)
            , Previous_(::GetEnv(name))
        {
            ::SetEnv(name, value);
        }

        ~TEnvGuard() {
            ::SetEnv(Name_, Previous_);
        }
    private:
        const TString Name_;
        const TString Previous_;
    };
}

TEST(Runtime, ArcadiaSourceRoot) {
    {
        auto tmpDir = ::GetSystemTempDir();
        TEnvGuard guard("ARCADIA_SOURCE_ROOT", tmpDir);
        EXPECT_EQ(tmpDir, ArcadiaSourceRoot());
    }
    {
        TEnvGuard guard("ARCADIA_SOURCE_ROOT", "");
        EXPECT_FALSE(ArcadiaSourceRoot().empty());
    }
}

TEST(Runtime, BuildRoot) {
    {
        auto tmpDir = ::GetSystemTempDir();
        TEnvGuard guard("ARCADIA_BUILD_ROOT", tmpDir);
        EXPECT_EQ(tmpDir, BuildRoot());
    }
    {
        TEnvGuard guard("ARCADIA_BUILD_ROOT", "");
        EXPECT_FALSE(BuildRoot().empty());
    }
}

TEST(Runtime, BinaryPath) {
    EXPECT_TRUE(TFsPath(BinaryPath("library/cpp/testing/common/ut")).Exists());
}

TEST(Runtime, GetArcadiaTestsData) {
    {
        auto tmpDir = ::GetSystemTempDir();
        TEnvGuard guard("ARCADIA_TESTS_DATA_DIR", tmpDir);
        EXPECT_EQ(tmpDir, GetArcadiaTestsData());
    }
    {
        TEnvGuard guard("ARCADIA_TESTS_DATA_DIR", "");
        auto path = GetArcadiaTestsData();
        // it is not error if path is empty
        const bool ok = (path.empty() || GetBaseName(path) == "arcadia_tests_data");
        EXPECT_TRUE(ok);
    }
}

TEST(Runtime, GetWorkPath) {
    {
        auto tmpDir = ::GetSystemTempDir();
        TEnvGuard guard("TEST_WORK_PATH", tmpDir);
        EXPECT_EQ(tmpDir, GetWorkPath());
    }
    {
        TEnvGuard guard("TEST_WORK_PATH", "");
        EXPECT_TRUE(!GetWorkPath().empty());
    }
}

TEST(Runtime, GetYaPath) {
    {
        auto tmpDir = ::GetSystemTempDir();
        TEnvGuard guard("YA_CACHE_DIR", tmpDir);
        EXPECT_EQ(tmpDir, GetYaPath().GetPath());
    }
    {
        TEnvGuard guard("YA_CACHE_DIR", "");
        EXPECT_EQ(GetYaPath().Basename(), ".ya");
    }
}

TEST(Runtime, GetOutputPath) {
    EXPECT_EQ(GetOutputPath().Basename(), "testing_out_stuff");
}

TEST(Runtime, GetRamDrivePath) {
    auto tmpDir = ::GetSystemTempDir();
    TEnvGuard guard("YA_TEST_RAM_DRIVE_PATH", tmpDir);
    EXPECT_EQ(tmpDir, GetRamDrivePath());
}

TEST(Runtime, GetOutputRamDrivePath) {
    auto tmpDir = ::GetSystemTempDir();
    TEnvGuard guard("YA_TEST_OUTPUT_RAM_DRIVE_PATH", tmpDir);
    EXPECT_EQ(tmpDir, GetOutputRamDrivePath());
}
