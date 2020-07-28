#include <library/cpp/testing/common/scope.h>

#include <util/system/env.h>

#include <library/cpp/testing/gtest/gtest.h>

TEST(TScopedEnvironment, SingleValue) {
    auto before = GetEnv("ARCADIA_SOURCE_ROOT");
    {
        NTesting::TScopedEnvironment guard("ARCADIA_SOURCE_ROOT", "source");
        EXPECT_EQ("source", GetEnv("ARCADIA_SOURCE_ROOT"));
    }
    EXPECT_EQ(before, GetEnv("ARCADIA_SOURCE_ROOT"));
}

TEST(TScopedEnvironment, MultiValue) {
    TVector<TString> before{GetEnv("ARCADIA_SOURCE_ROOT"), GetEnv("ARCADIA_BUILD_ROOT")};
    {
        NTesting::TScopedEnvironment guard{{
            {"ARCADIA_SOURCE_ROOT", "source"},
            {"ARCADIA_BUILD_ROOT", "build"},
        }};
        EXPECT_EQ("source", GetEnv("ARCADIA_SOURCE_ROOT"));
        EXPECT_EQ("build", GetEnv("ARCADIA_BUILD_ROOT"));
    }
    TVector<TString> after{GetEnv("ARCADIA_SOURCE_ROOT"), GetEnv("ARCADIA_BUILD_ROOT")};
    EXPECT_EQ(before, after);
}
