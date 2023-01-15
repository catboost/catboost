#include "yarchive.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/string/cast.h>
#include <util/stream/file.h>
#include <util/system/tempfile.h>
#include <util/memory/blob.h>

class TArchiveTest: public TTestBase {
    UNIT_TEST_SUITE(TArchiveTest)
    UNIT_TEST(TestCreate);
    UNIT_TEST(TestRead);
    UNIT_TEST(TestOffsetOrder);
    UNIT_TEST_SUITE_END();

private:
    void CreateArchive();
    void TestCreate();
    void TestRead();
    void TestOffsetOrder();
};

UNIT_TEST_SUITE_REGISTRATION(TArchiveTest);

#define ARCHIVE "./test.ar"

void TArchiveTest::CreateArchive() {
    TFixedBufferFileOutput out(ARCHIVE);
    TArchiveWriter w(&out);

    for (size_t i = 0; i < 1000; ++i) {
        const TString path = "/" + ToString(i);
        const TString data = "data" + ToString(i * 1000) + "dataend";
        TStringInput si(data);

        w.Add(path, &si);
    }

    w.Finish();
    out.Finish();
}

void TArchiveTest::TestCreate() {
    CreateArchive();
    TTempFile tmpFile(ARCHIVE);
}

void TArchiveTest::TestRead() {
    CreateArchive();
    TTempFile tmpFile(ARCHIVE);
    TBlob blob = TBlob::FromFileSingleThreaded(ARCHIVE);
    TArchiveReader r(blob);

    UNIT_ASSERT_EQUAL(r.Count(), 1000);

    for (size_t i = 0; i < 1000; ++i) {
        const TString key = "/" + ToString(i);
        TAutoPtr<IInputStream> is = r.ObjectByKey(key);
        const TString data = is->ReadAll();

        UNIT_ASSERT_EQUAL(data, "data" + ToString(i * 1000) + "dataend");
    }
}

void TArchiveTest::TestOffsetOrder() {
    CreateArchive();
    TTempFile tmpFile(ARCHIVE);
    TBlob blob1 = TBlob::FromFileSingleThreaded(ARCHIVE);
    TArchiveReader r(blob1);

    const void* prevOffset = nullptr;

    for (size_t i = 0; i < r.Count(); ++i) {
        const TString key = r.KeyByIndex(i);
        TBlob blob2 = r.BlobByKey(key);
        const void* offset = blob2.Data();

        if (i) {
            UNIT_ASSERT(prevOffset < offset);
        }
        prevOffset = offset;
    }
}
