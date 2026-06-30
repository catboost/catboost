#include "src_root.h"

#include <util/folder/pathsplit.h>
#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(TestSourceRoot) {
    Y_UNIT_TEST(TestStrip) {
        // Reconstruct() converts "\" -> "/" on Windows
        const TString path = TPathSplit(__SOURCE_FILE_IMPL__.As<TStringBuf>()).Reconstruct();
        UNIT_ASSERT_EQUAL(path, "util" LOCSLASH_S "system" LOCSLASH_S "src_root_ut.cpp");
    }

    Y_UNIT_TEST(TestPrivateChopPrefixRoutine) {
        static constexpr const char str[] = ":\0:\0: It's unlikely that this string has an ARCADIA_ROOT as its prefix :\0:\0:";
        static constexpr const auto strStaticBuf = STATIC_BUF(str);
        UNIT_ASSERT_VALUES_EQUAL(TStringBuf(str, sizeof(str) - 1), ::NPrivate::StripRoot(strStaticBuf).As<TStringBuf>());
        UNIT_ASSERT_VALUES_EQUAL(0, ::NPrivate::RootPrefixLength(strStaticBuf));

        static_assert(::NPrivate::IsProperPrefix(STATIC_BUF("foo"), STATIC_BUF("foobar")), R"(IsProperPrefix("foo", "foobar") failed)");
        static_assert(!::NPrivate::IsProperPrefix(STATIC_BUF("foobar"), STATIC_BUF("foo")), R"(IsProperPrefix("foobar", "foo") failed)");
        static_assert(!::NPrivate::IsProperPrefix(STATIC_BUF("name"), STATIC_BUF("name")), R"(IsProperPrefix("name", "name") failed)");
        static_assert(::NPrivate::IsProperPrefix(STATIC_BUF("name"), STATIC_BUF("name/")), R"(IsProperPrefix("name", "name/") failed)");
        static_assert(::NPrivate::IsProperPrefix(STATIC_BUF(""), STATIC_BUF("foobar")), R"(IsProperPrefix("", "foobar") failed)");
        static_assert(!::NPrivate::IsProperPrefix(STATIC_BUF(""), STATIC_BUF("")), R"(IsProperPrefix("", "") failed)");
        static_assert(::NPrivate::IsProperPrefix(STATIC_BUF("dir"), STATIC_BUF("dir/file")), R"(IsProperPrefix("dir", "dir/file") failed)");
    }
} // Y_UNIT_TEST_SUITE(TestSourceRoot)
