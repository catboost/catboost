#include <library/cpp/testing/common/env.h>
#include <library/cpp/testing/common/scope.h>
#include <library/cpp/testing/gtest/gtest.h>

#include <util/folder/dirut.h>
#include <util/stream/file.h>
#include <util/system/env.h>
#include <util/system/execpath.h>
#include <util/system/fs.h>


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

#ifdef _linux_
TEST(Runtime, GdbPath) {
    Singleton<NPrivate::TTestEnv>()->ReInitialize();
    EXPECT_TRUE(NFs::Exists(::GdbPath()));
}
#endif

TString ReInitializeContext(TStringBuf data) {
    auto tmpDir = ::GetSystemTempDir();
    auto filename = tmpDir + "/context.json";
    TOFStream stream(filename);
    stream.Write(data.data(), data.size());
    stream.Finish();

    NTesting::TScopedEnvironment contextGuard("YA_TEST_CONTEXT_FILE", filename);
    Singleton<NPrivate::TTestEnv>()->ReInitialize();

    return filename;
}

TEST(Runtime, GetTestParam) {
    TString context = R"json({
        "runtime": {
            "test_params": {
                "a": "b",
                "c": "d"
            }
        }
    })json";
    auto filename = ReInitializeContext(context);

    EXPECT_EQ("b", GetTestParam("a"));
    EXPECT_EQ("d", GetTestParam("c"));
    EXPECT_EQ("", GetTestParam("e"));
    EXPECT_EQ("w", GetTestParam("e", "w"));

    Singleton<NPrivate::TTestEnv>()->AddTestParam("e", "e");
    EXPECT_EQ("e", GetTestParam("e"));
}

TEST(Runtime, WatchProcessCore) {
    TString context = R"json({
        "internal": {
            "core_search_file": "watch_core.txt"
        }
    })json";
    auto filename = ReInitializeContext(context);

    WatchProcessCore(1, "bin1", "pwd");
    WatchProcessCore(2, "bin1");
    StopProcessCoreWatching(2);

    TIFStream file("watch_core.txt");
    auto data = file.ReadAll();
    TString expected = R"json({"cmd":"add","pid":1,"binary_path":"bin1","cwd":"pwd"}
{"cmd":"add","pid":2,"binary_path":"bin1"}
{"cmd":"drop","pid":2}
)json";
    EXPECT_EQ(expected, data);
}

TEST(Runtime, GlobalResources) {
    TString context = R"json({
        "resources": {
            "global": {
                "TOOL_NAME_RESOURCE_GLOBAL": "path"
            }
        }
    })json";

    auto filename = ReInitializeContext(context);
    EXPECT_EQ("path", GetGlobalResource("TOOL_NAME_RESOURCE_GLOBAL"));
}
