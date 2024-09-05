#include "blob.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/system/tempfile.h>
#include <util/folder/path.h>
#include <util/stream/output.h>
#include <util/stream/file.h>
#include <util/generic/buffer.h>
#include <util/generic/array_ref.h>

Y_UNIT_TEST_SUITE(TBlobTest) {
    Y_UNIT_TEST(TestSubBlob) {
        TBlob child;
        const char* p = nullptr;

        {
            TBlob parent = TBlob::CopySingleThreaded("0123456789", 10);
            UNIT_ASSERT_EQUAL(parent.Length(), 10);
            p = parent.AsCharPtr();
            UNIT_ASSERT_EQUAL(memcmp(p, "0123456789", 10), 0);
            child = parent.SubBlob(2, 5);
        } // Don't worry about parent

        UNIT_ASSERT_EQUAL(child.Length(), 3);
        UNIT_ASSERT_EQUAL(memcmp(child.AsCharPtr(), "234", 3), 0);
        UNIT_ASSERT_EQUAL(p + 2, child.AsCharPtr());
    }

    Y_UNIT_TEST(TestFromStream) {
        TString s("sjklfgsdyutfuyas54fa78s5f89a6df790asdf7");
        TMemoryInput mi(s.data(), s.size());
        TBlob b = TBlob::FromStreamSingleThreaded(mi);

        UNIT_ASSERT_EQUAL(TString((const char*)b.Data(), b.Length()), s);
    }

    Y_UNIT_TEST(TestFromString) {
        TString s("dsfkjhgsadftusadtf");
        TBlob b(TBlob::FromString(s));

        UNIT_ASSERT_EQUAL(TString((const char*)b.Data(), b.Size()), s);
        const auto expectedRef = TArrayRef<const ui8>{(ui8*)s.data(), s.size()};
        UNIT_ASSERT_EQUAL(TArrayRef<const ui8>{b}, expectedRef);
    }

    Y_UNIT_TEST(TestFromBuffer) {
        const size_t sz = 1234u;
        TBuffer buf;
        buf.Resize(sz);
        UNIT_ASSERT_EQUAL(buf.Size(), sz);
        TBlob b = TBlob::FromBuffer(buf);
        UNIT_ASSERT_EQUAL(buf.Size(), 0u);
        UNIT_ASSERT_EQUAL(b.Size(), sz);
    }

    Y_UNIT_TEST(TestFromFile) {
        TString path = "testfile";

        TOFStream stream(path);
        stream.Write("1234", 4);
        stream.Finish();

        auto testMode = [](TBlob blob) {
            UNIT_ASSERT_EQUAL(blob.Size(), 4);
            UNIT_ASSERT_EQUAL(TStringBuf(static_cast<const char*>(blob.Data()), 4), "1234");
        };

        testMode(TBlob::FromFile(path));
        testMode(TBlob::PrechargedFromFile(path));
        testMode(TBlob::LockedFromFile(path));
    }

    Y_UNIT_TEST(TestEmptyLockedFiles) {
        TString path = MakeTempName();
        TFsPath(path).Touch();
        TBlob::LockedFromFile(path);
    }
} // Y_UNIT_TEST_SUITE(TBlobTest)
