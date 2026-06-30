// File includes itself to make multiple passes of its suites with different platform-dependent definitions

#ifndef PS_INCLUDED
// Outer part

    #include "pathsplit.h"

    #include <library/cpp/testing/unittest/registar.h>

    #define VAR(NAME) Y_CAT(NAME, __LINE__)

    #define PS_CHECK(input, ...)                                                       \
        const char* VAR(model)[] = {"", __VA_ARGS__};                                  \
        UNIT_ASSERT_EQUAL(input.size(), sizeof(VAR(model)) / sizeof(const char*) - 1); \
        for (size_t n = 0; n < input.size(); ++n) {                                    \
            UNIT_ASSERT_STRINGS_EQUAL(input[n], VAR(model)[n + 1]);                    \
        }

    #define PS_INCLUDED

    #define PSUF(NAME) NAME
    #define PSUF_LOCAL(NAME) NAME##Local
    #include <util/folder/pathsplit_ut.cpp>
    #undef PSUF
    #undef PSUF_LOCAL

    #define PSUF(NAME) NAME##Unix
    #define PSUF_LOCAL(NAME) PSUF(NAME)
    #ifdef _win_
        #undef _win_
        #define REVERT_WIN
    #endif
    #include <util/folder/pathsplit_ut.cpp>
    #ifdef REVERT_WIN
        #define _win_
        #undef REVERT_WIN
    #endif
    #undef PSUF
    #undef PSUF_LOCAL

    #define PSUF(NAME) NAME##Windows
    #define PSUF_LOCAL(NAME) PSUF(NAME)
    #ifndef _win_
        #define _win_
        #define REVERT_WIN
    #endif
    #include <util/folder/pathsplit_ut.cpp>
    #ifdef REVERT_WIN
        #undef _win_
        #undef REVERT_WIN
    #endif
    #undef PSUF
    #undef PSUF_LOCAL

    #undef PS_INCLUDED

#else
// Inner part

    #ifdef _win_
        #define TRUE_ONLY_WIN true
    #else
        #define TRUE_ONLY_WIN false
    #endif

Y_UNIT_TEST_SUITE(PSUF(PathSplit)) {
    Y_UNIT_TEST(Empty) {
        PSUF(TPathSplit)
        ps;
        PS_CHECK(ps);
        UNIT_ASSERT_EQUAL(ps.IsAbsolute, false);
    }

    Y_UNIT_TEST(Relative) {
        PSUF(TPathSplit)
        ps("some/usual/path");
        PS_CHECK(ps, "some", "usual", "path");
        UNIT_ASSERT_EQUAL(ps.IsAbsolute, false);
    }

    Y_UNIT_TEST(Absolute) {
        PSUF(TPathSplit)
        ps("/some/usual/path");
        PS_CHECK(ps, "some", "usual", "path");
        UNIT_ASSERT_EQUAL(ps.IsAbsolute, true);
    }

    Y_UNIT_TEST(Self) {
        PSUF(TPathSplit)
        ps(".");
        PS_CHECK(ps, ".");
        UNIT_ASSERT_EQUAL(ps.IsAbsolute, false);
    }

    Y_UNIT_TEST(Parent) {
        PSUF(TPathSplit)
        ps("..");
        PS_CHECK(ps, "..");
        UNIT_ASSERT_EQUAL(ps.IsAbsolute, false);
    }

    Y_UNIT_TEST(Root) {
        PSUF(TPathSplit)
        ps("/");
        PS_CHECK(ps);
        UNIT_ASSERT_EQUAL(ps.IsAbsolute, true);
    }

    Y_UNIT_TEST(Reconstruct) {
        PSUF(TPathSplit)
        ps("some/usual/path/../../other/././//path");
    #ifdef _win_
        UNIT_ASSERT_STRINGS_EQUAL(ps.Reconstruct(), "some\\other\\path");
    #else
        UNIT_ASSERT_STRINGS_EQUAL(ps.Reconstruct(), "some/other/path");
    #endif

        ps = PSUF(TPathSplit)("/some/usual/path/../../other/././//path");
    #ifdef _win_
        UNIT_ASSERT_STRINGS_EQUAL(ps.Reconstruct(), "\\some\\other\\path");
    #else
        UNIT_ASSERT_STRINGS_EQUAL(ps.Reconstruct(), "/some/other/path");
    #endif
    }

    Y_UNIT_TEST(ParseFirstPart) {
        PSUF(TPathSplit)
        ps;
        ps.ParseFirstPart("some/usual/path");
        PS_CHECK(ps, "some", "usual", "path");
        UNIT_ASSERT_EQUAL(ps.IsAbsolute, false);

        ps = PSUF(TPathSplit)();
        ps.ParseFirstPart("/some/usual/path");
        PS_CHECK(ps, "some", "usual", "path");
        UNIT_ASSERT_EQUAL(ps.IsAbsolute, true);
    }

    Y_UNIT_TEST(ParsePart) {
        PSUF(TPathSplit)
        ps("some/usual/path");
        ps.ParsePart("sub/path");
        PS_CHECK(ps, "some", "usual", "path", "sub", "path");
        UNIT_ASSERT_EQUAL(ps.IsAbsolute, false);

        ps = PSUF(TPathSplit)("some/usual/path");
        ps.ParsePart("/sub/path");
        PS_CHECK(ps, "some", "usual", "path", "sub", "path");
        UNIT_ASSERT_EQUAL(ps.IsAbsolute, false);
    }

    Y_UNIT_TEST(ParsePartSelf) {
        PSUF(TPathSplit)
        ps("some/usual/path");
        ps.ParsePart(".");
        PS_CHECK(ps, "some", "usual", "path");

        ps = PSUF(TPathSplit)("some/usual/path");
        ps.ParsePart("././.");
        PS_CHECK(ps, "some", "usual", "path");
    }

    Y_UNIT_TEST(ParsePartParent) {
        PSUF(TPathSplit)
        ps("some/usual/path");
        ps.ParsePart("..");
        PS_CHECK(ps, "some", "usual");

        ps = PSUF(TPathSplit)("some/usual/path");
        ps.ParsePart("../..");
        PS_CHECK(ps, "some");

        ps = PSUF(TPathSplit)("some/usual/path");
        ps.ParsePart("../../..");
        PS_CHECK(ps);
        UNIT_ASSERT_EQUAL(ps.IsAbsolute, false);

        ps = PSUF(TPathSplit)("/some/usual/path");
        ps.ParsePart("../../..");
        PS_CHECK(ps);
        UNIT_ASSERT_EQUAL(ps.IsAbsolute, true);
    }

    Y_UNIT_TEST(ParsePartOverflow) {
        PSUF(TPathSplit)
        ps("some/usual/path");
        ps.ParsePart("../../../../..");
        PS_CHECK(ps, "..", "..");
        UNIT_ASSERT_EQUAL(ps.IsAbsolute, false);

        ps = PSUF(TPathSplit)("/some/usual/path");
        ps.ParsePart("../../../../..");
        PS_CHECK(ps, "..", "..");
        UNIT_ASSERT_EQUAL(ps.IsAbsolute, true);
    }

    Y_UNIT_TEST(WinRelative) {
        PSUF(TPathSplit)
        ps("some\\usual\\path");
    #ifdef _win_
        PS_CHECK(ps, "some", "usual", "path");
    #else
        PS_CHECK(ps, "some\\usual\\path");
    #endif
        UNIT_ASSERT_EQUAL(ps.IsAbsolute, false);
    }

    Y_UNIT_TEST(WinAbsolute) {
        PSUF(TPathSplit)
        ps("\\some\\usual\\path");
    #ifdef _win_
        PS_CHECK(ps, "some", "usual", "path");
    #else
        PS_CHECK(ps, "\\some\\usual\\path");
    #endif
        UNIT_ASSERT_EQUAL(ps.IsAbsolute, TRUE_ONLY_WIN);

        PSUF(TPathSplit)
        psDrive("C:\\some\\usual\\path");
    #ifdef _win_
        PS_CHECK(psDrive, "some", "usual", "path");
        UNIT_ASSERT_EQUAL(psDrive.Drive, "C:");
    #else
        PS_CHECK(psDrive, "C:\\some\\usual\\path");
    #endif
        UNIT_ASSERT_EQUAL(psDrive.IsAbsolute, TRUE_ONLY_WIN);

        PSUF(TPathSplit)
        psDrive2("C:/some/usual/path");
    #ifdef _win_
        PS_CHECK(psDrive2, "some", "usual", "path");
        UNIT_ASSERT_EQUAL(psDrive2.Drive, "C:");
    #else
        PS_CHECK(psDrive2, "C:", "some", "usual", "path");
    #endif
        UNIT_ASSERT_EQUAL(psDrive2.IsAbsolute, TRUE_ONLY_WIN);
    }

    Y_UNIT_TEST(WinRoot) {
        PSUF(TPathSplit)
        ps("\\");
    #ifdef _win_
        PS_CHECK(ps);
    #else
        PS_CHECK(ps, "\\");
    #endif
        UNIT_ASSERT_EQUAL(ps.IsAbsolute, TRUE_ONLY_WIN);

        PSUF(TPathSplit)
        psDrive("C:");
    #ifdef _win_
        PS_CHECK(psDrive);
        UNIT_ASSERT_EQUAL(psDrive.Drive, "C:");
    #else
        PS_CHECK(psDrive, "C:");
    #endif
        UNIT_ASSERT_EQUAL(psDrive.IsAbsolute, TRUE_ONLY_WIN);
    }

    Y_UNIT_TEST(WinReconstruct) {
        PSUF(TPathSplit)
        ps("some\\usual\\path\\..\\..\\other\\.\\.\\\\\\path");
    #ifdef _win_
        UNIT_ASSERT_STRINGS_EQUAL(ps.Reconstruct(), "some\\other\\path");
    #else
        UNIT_ASSERT_STRINGS_EQUAL(ps.Reconstruct(), "some\\usual\\path\\..\\..\\other\\.\\.\\\\\\path");
    #endif

        ps = PSUF(TPathSplit)("\\some\\usual\\path\\..\\..\\other\\.\\.\\\\\\path");
    #ifdef _win_
        UNIT_ASSERT_STRINGS_EQUAL(ps.Reconstruct(), "\\some\\other\\path");
    #else
        UNIT_ASSERT_STRINGS_EQUAL(ps.Reconstruct(), "\\some\\usual\\path\\..\\..\\other\\.\\.\\\\\\path");
    #endif
    }

    Y_UNIT_TEST(WinParseFirstPart) {
        PSUF(TPathSplit)
        ps;
        ps.ParseFirstPart("some\\usual\\path");
    #ifdef _win_
        PS_CHECK(ps, "some", "usual", "path");
    #else
        PS_CHECK(ps, "some\\usual\\path");
    #endif
        UNIT_ASSERT_EQUAL(ps.IsAbsolute, false);

        ps = PSUF(TPathSplit)();
        ps.ParseFirstPart("\\some\\usual\\path");
    #ifdef _win_
        PS_CHECK(ps, "some", "usual", "path");
    #else
        PS_CHECK(ps, "\\some\\usual\\path");
    #endif
        UNIT_ASSERT_EQUAL(ps.IsAbsolute, TRUE_ONLY_WIN);
    }

    Y_UNIT_TEST(WinParsePart) {
        PSUF(TPathSplit)
        ps("some\\usual\\path");
        ps.ParsePart("sub\\path");
    #ifdef _win_
        PS_CHECK(ps, "some", "usual", "path", "sub", "path");
    #else
        PS_CHECK(ps, "some\\usual\\path", "sub\\path");
    #endif
        UNIT_ASSERT_EQUAL(ps.IsAbsolute, false);

        ps = PSUF(TPathSplit)("some\\usual\\path");
        ps.ParsePart("\\sub\\path");
    #ifdef _win_
        PS_CHECK(ps, "some", "usual", "path", "sub", "path");
    #else
        PS_CHECK(ps, "some\\usual\\path", "\\sub\\path");
    #endif
        UNIT_ASSERT_EQUAL(ps.IsAbsolute, false);
    }

    #ifdef _win_
    Y_UNIT_TEST(WinParsePartSelf) {
        PSUF(TPathSplit)
        ps("some\\usual\\path");
        ps.ParsePart(".");
        PS_CHECK(ps, "some", "usual", "path");

        ps = PSUF(TPathSplit)("some\\usual\\path");
        ps.ParsePart(".\\.\\.");
        PS_CHECK(ps, "some", "usual", "path");
    }

    Y_UNIT_TEST(WinParsePartParent) {
        PSUF(TPathSplit)
        ps("some\\usual\\path");
        ps.ParsePart("..");
        PS_CHECK(ps, "some", "usual");

        ps = PSUF(TPathSplit)("some\\usual\\path");
        ps.ParsePart("..\\..");
        PS_CHECK(ps, "some");

        ps = PSUF(TPathSplit)("some\\usual\\path");
        ps.ParsePart("..\\..\\..");
        PS_CHECK(ps);
        UNIT_ASSERT_EQUAL(ps.IsAbsolute, false);

        ps = PSUF(TPathSplit)("\\some\\usual\\path");
        ps.ParsePart("..\\..\\..");
        PS_CHECK(ps);
        UNIT_ASSERT_EQUAL(ps.IsAbsolute, true);

        ps = PSUF(TPathSplit)("C:\\some\\usual\\path");
        ps.ParsePart("..\\..\\..");
        PS_CHECK(ps);
        UNIT_ASSERT_EQUAL(ps.IsAbsolute, true);
        UNIT_ASSERT_EQUAL(ps.Drive, "C:");
    }

    Y_UNIT_TEST(WinParsePartOverflow) {
        PSUF(TPathSplit)
        ps("some\\usual\\path");
        ps.ParsePart("..\\..\\..\\..\\..");
        PS_CHECK(ps, "..", "..");
        UNIT_ASSERT_EQUAL(ps.IsAbsolute, false);

        ps = PSUF(TPathSplit)("\\some\\usual\\path");
        ps.ParsePart("..\\..\\..\\..\\..");
        PS_CHECK(ps, "..", "..");
        UNIT_ASSERT_EQUAL(ps.IsAbsolute, true);

        ps = PSUF(TPathSplit)("C:\\some\\usual\\path");
        ps.ParsePart("..\\..\\..\\..\\..");
        PS_CHECK(ps, "..", "..");
        UNIT_ASSERT_EQUAL(ps.IsAbsolute, true);
        UNIT_ASSERT_EQUAL(ps.Drive, "C:");
    }
    #endif

    Y_UNIT_TEST(WinMixed) {
        PSUF(TPathSplit)
        ps("some\\usual/path");
    #ifdef _win_
        PS_CHECK(ps, "some", "usual", "path");
    #else
        PS_CHECK(ps, "some\\usual", "path");
    #endif
        UNIT_ASSERT_EQUAL(ps.IsAbsolute, false);
    }

    Y_UNIT_TEST(WinParsePartMixed) {
        PSUF(TPathSplit)
        ps("some\\usual/path");
        ps.ParsePart("sub/sub\\path");
    #ifdef _win_
        PS_CHECK(ps, "some", "usual", "path", "sub", "sub", "path");
    #else
        PS_CHECK(ps, "some\\usual", "path", "sub", "sub\\path");
    #endif
        UNIT_ASSERT_EQUAL(ps.IsAbsolute, false);
    }

    Y_UNIT_TEST(BeginWithSelf) {
        PSUF(TPathSplit)
        ps("./some/usual/path");
        PS_CHECK(ps, "some", "usual", "path");
    #ifdef _win_
        UNIT_ASSERT_STRINGS_EQUAL(ps.Reconstruct(), "some\\usual\\path");
    #else
        UNIT_ASSERT_STRINGS_EQUAL(ps.Reconstruct(), "some/usual/path");
    #endif
    }

    Y_UNIT_TEST(BeginWithParent) {
        PSUF(TPathSplit)
        ps("../some/usual/path");
        PS_CHECK(ps, "..", "some", "usual", "path");
    #ifdef _win_
        UNIT_ASSERT_STRINGS_EQUAL(ps.Reconstruct(), "..\\some\\usual\\path");
    #else
        UNIT_ASSERT_STRINGS_EQUAL(ps.Reconstruct(), "../some/usual/path");
    #endif
    }

    Y_UNIT_TEST(InOut) {
        PSUF(TPathSplit)
        ps("path/..");
        PS_CHECK(ps);
        UNIT_ASSERT_STRINGS_EQUAL(ps.Reconstruct(), "");
    }

    Y_UNIT_TEST(OutIn) {
        PSUF(TPathSplit)
        ps("../path");
        PS_CHECK(ps, "..", "path");
    #ifdef _win_
        UNIT_ASSERT_STRINGS_EQUAL(ps.Reconstruct(), "..\\path");
    #else
        UNIT_ASSERT_STRINGS_EQUAL(ps.Reconstruct(), "../path");
    #endif
    }
} // Y_UNIT_TEST_SUITE(PSUF(PathSplit)

Y_UNIT_TEST_SUITE(PSUF(PathSplitTraits)) {
    Y_UNIT_TEST(IsPathSep) {
        UNIT_ASSERT_EQUAL(PSUF_LOCAL(TPathSplitTraits)::IsPathSep('/'), true);
        UNIT_ASSERT_EQUAL(PSUF_LOCAL(TPathSplitTraits)::IsPathSep('\\'), TRUE_ONLY_WIN);
        UNIT_ASSERT_EQUAL(PSUF_LOCAL(TPathSplitTraits)::IsPathSep(' '), false);
    }

    Y_UNIT_TEST(IsAbsolutePath) {
        UNIT_ASSERT_EQUAL(PSUF_LOCAL(TPathSplitTraits)::IsAbsolutePath(""), false);
        UNIT_ASSERT_EQUAL(PSUF_LOCAL(TPathSplitTraits)::IsAbsolutePath("/"), true);
        UNIT_ASSERT_EQUAL(PSUF_LOCAL(TPathSplitTraits)::IsAbsolutePath("some/usual/path"), false);
        UNIT_ASSERT_EQUAL(PSUF_LOCAL(TPathSplitTraits)::IsAbsolutePath("/some/usual/path"), true);
        UNIT_ASSERT_EQUAL(PSUF_LOCAL(TPathSplitTraits)::IsAbsolutePath("."), false);
        UNIT_ASSERT_EQUAL(PSUF_LOCAL(TPathSplitTraits)::IsAbsolutePath(".."), false);
        UNIT_ASSERT_EQUAL(PSUF_LOCAL(TPathSplitTraits)::IsAbsolutePath("/."), true);
        UNIT_ASSERT_EQUAL(PSUF_LOCAL(TPathSplitTraits)::IsAbsolutePath("/.."), true);
    }

    Y_UNIT_TEST(WinIsAbsolutePath) {
        UNIT_ASSERT_EQUAL(PSUF_LOCAL(TPathSplitTraits)::IsAbsolutePath("somepath"), false);
        UNIT_ASSERT_EQUAL(PSUF_LOCAL(TPathSplitTraits)::IsAbsolutePath("\\"), TRUE_ONLY_WIN);
        UNIT_ASSERT_EQUAL(PSUF_LOCAL(TPathSplitTraits)::IsAbsolutePath("\\somepath"), TRUE_ONLY_WIN);
        UNIT_ASSERT_EQUAL(PSUF_LOCAL(TPathSplitTraits)::IsAbsolutePath("\\."), TRUE_ONLY_WIN);
        UNIT_ASSERT_EQUAL(PSUF_LOCAL(TPathSplitTraits)::IsAbsolutePath("\\.."), TRUE_ONLY_WIN);
        UNIT_ASSERT_EQUAL(PSUF_LOCAL(TPathSplitTraits)::IsAbsolutePath("C"), false);
        UNIT_ASSERT_EQUAL(PSUF_LOCAL(TPathSplitTraits)::IsAbsolutePath("C:"), TRUE_ONLY_WIN);
        UNIT_ASSERT_EQUAL(PSUF_LOCAL(TPathSplitTraits)::IsAbsolutePath("C:somepath"), false);
        UNIT_ASSERT_EQUAL(PSUF_LOCAL(TPathSplitTraits)::IsAbsolutePath("C:\\"), TRUE_ONLY_WIN);
        UNIT_ASSERT_EQUAL(PSUF_LOCAL(TPathSplitTraits)::IsAbsolutePath("C:\\somepath"), TRUE_ONLY_WIN);
        UNIT_ASSERT_EQUAL(PSUF_LOCAL(TPathSplitTraits)::IsAbsolutePath("C:/"), TRUE_ONLY_WIN);
        UNIT_ASSERT_EQUAL(PSUF_LOCAL(TPathSplitTraits)::IsAbsolutePath("C:/somepath"), TRUE_ONLY_WIN);
        UNIT_ASSERT_EQUAL(PSUF_LOCAL(TPathSplitTraits)::IsAbsolutePath("#:"), false);
        UNIT_ASSERT_EQUAL(PSUF_LOCAL(TPathSplitTraits)::IsAbsolutePath("#:somepath"), false);
        UNIT_ASSERT_EQUAL(PSUF_LOCAL(TPathSplitTraits)::IsAbsolutePath("#:\\somepath"), false);
        UNIT_ASSERT_EQUAL(PSUF_LOCAL(TPathSplitTraits)::IsAbsolutePath("#:/somepath"), false);
    }
} // Y_UNIT_TEST_SUITE(PSUF(PathSplitTraits)

    #undef TRUE_ONLY_WIN

#endif
