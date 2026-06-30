#include "tempfile.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/folder/dirut.h>
#include <util/generic/yexception.h>
#include <util/stream/file.h>

#include <algorithm>

Y_UNIT_TEST_SUITE(TTempFileHandle) {
    Y_UNIT_TEST(Create) {
        TString path;
        {
            TTempFileHandle tmp;
            path = tmp.Name();
            tmp.Write("hello world\n", 12);
            tmp.FlushData();
            UNIT_ASSERT_STRINGS_EQUAL(TUnbufferedFileInput(tmp.Name()).ReadAll(), "hello world\n");
        }
        UNIT_ASSERT(!NFs::Exists(path));
    }

    Y_UNIT_TEST(InCurrentDir) {
#ifndef _win32_
        static const TString TEST_PREFIX = "unique_prefix";
#else
        static const TString TEST_PREFIX = "uni";
#endif

        TString path;
        {
            TTempFileHandle tmp = TTempFileHandle::InCurrentDir(TEST_PREFIX);
            path = tmp.Name();
            UNIT_ASSERT(NFs::Exists(path));

            TVector<TString> names;
            TFsPath(".").ListNames(names);
            bool containsFileWithPrefix = std::any_of(names.begin(), names.end(), [&](const TString& name) {
                return name.Contains(TEST_PREFIX);
            });
            UNIT_ASSERT(containsFileWithPrefix);
        }
        UNIT_ASSERT(!NFs::Exists(path));
    }

    Y_UNIT_TEST(UseExtensionWithoutDot) {
        TString path;
        {
            TTempFileHandle tmp = TTempFileHandle::InCurrentDir("hello", "world");
            path = tmp.Name();
            UNIT_ASSERT(NFs::Exists(path));

#ifndef _win32_
            UNIT_ASSERT(path.Contains("hello"));
            UNIT_ASSERT(path.EndsWith(".world"));
            UNIT_ASSERT(!path.EndsWith("..world"));
#else
            UNIT_ASSERT(path.Contains("hel"));
            UNIT_ASSERT(path.EndsWith(".tmp"));
#endif
        }
        UNIT_ASSERT(!NFs::Exists(path));
    }

    Y_UNIT_TEST(UseExtensionWithDot) {
        TString path;
        {
            TTempFileHandle tmp = TTempFileHandle::InCurrentDir("lorem", ".ipsum");
            path = tmp.Name();
            UNIT_ASSERT(NFs::Exists(path));

#ifndef _win32_
            UNIT_ASSERT(path.Contains("lorem"));
            UNIT_ASSERT(path.EndsWith(".ipsum"));
            UNIT_ASSERT(!path.EndsWith("..ipsum"));
#else
            UNIT_ASSERT(path.Contains("lor"));
            UNIT_ASSERT(path.EndsWith(".tmp"));
#endif
        }
        UNIT_ASSERT(!NFs::Exists(path));
    }

    Y_UNIT_TEST(SafeDestructor) {
        TString path;
        {
            path = MakeTempName();
            UNIT_ASSERT(NFs::Exists(path));

            TTempFileHandle tmp(path);
            Y_UNUSED(tmp);
            UNIT_ASSERT(NFs::Exists(path));

            TTempFileHandle anotherTmp(path);
            Y_UNUSED(anotherTmp);
            UNIT_ASSERT(NFs::Exists(path));
        }
        UNIT_ASSERT(!NFs::Exists(path));
    }

    Y_UNIT_TEST(RemovesOpen) {
        TString path;
        {
            TTempFileHandle tmp;
            path = tmp.Name();
            tmp.Write("hello world\n", 12);
            tmp.FlushData();
            UNIT_ASSERT(NFs::Exists(path));
            UNIT_ASSERT(tmp.IsOpen());
        }
        UNIT_ASSERT(!NFs::Exists(path));
    }

    Y_UNIT_TEST(NonExistingDirectory) {
        UNIT_ASSERT_EXCEPTION(TTempFileHandle::InDir("nonexsistingdirname"), TSystemError);
    }
} // Y_UNIT_TEST_SUITE(TTempFileHandle)

Y_UNIT_TEST_SUITE(MakeTempName) {
    Y_UNIT_TEST(Default) {
        TString path;
        {
            TTempFile tmp(MakeTempName());
            path = tmp.Name();

            UNIT_ASSERT(!path.Contains('\0'));
            UNIT_ASSERT(NFs::Exists(path));
            UNIT_ASSERT(path.EndsWith(".tmp"));

#ifndef _win32_
            UNIT_ASSERT(path.Contains("yandex"));
#else
            UNIT_ASSERT(path.Contains("yan"));
#endif
        }
        UNIT_ASSERT(!NFs::Exists(path));
    }

    Y_UNIT_TEST(UseNullptr) {
        TString path;
        {
            TTempFile tmp(MakeTempName(nullptr, nullptr, nullptr));
            path = tmp.Name();

            UNIT_ASSERT(!path.Contains('\0'));
            UNIT_ASSERT(NFs::Exists(path));
        }
        UNIT_ASSERT(!NFs::Exists(path));
    }
} // Y_UNIT_TEST_SUITE(MakeTempName)
