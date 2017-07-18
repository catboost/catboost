#include "blob.h"

#include <library/unittest/registar.h>

#include <util/stream/output.h>
#include <util/generic/buffer.h>

SIMPLE_UNIT_TEST_SUITE(TBlobTest){
    SIMPLE_UNIT_TEST(TestSubBlob){
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

SIMPLE_UNIT_TEST(TestFromStream) {
    TString s("sjklfgsdyutfuyas54fa78s5f89a6df790asdf7");
    TMemoryInput mi(~s, +s);
    TBlob b = TBlob::FromStreamSingleThreaded(mi);

    UNIT_ASSERT_EQUAL(TString((const char*)b.Data(), b.Length()), s);
}

SIMPLE_UNIT_TEST(TestFromString) {
    TString s("dsfkjhgsadftusadtf");
    TBlob b(TBlob::FromString(s));

    UNIT_ASSERT_EQUAL(TString((const char*)b.Data(), b.Size()), s);
}

SIMPLE_UNIT_TEST(TestFromBuffer) {
    const size_t sz = 1234u;
    TBuffer buf;
    buf.Resize(sz);
    UNIT_ASSERT_EQUAL(buf.Size(), sz);
    TBlob b = TBlob::FromBuffer(buf);
    UNIT_ASSERT_EQUAL(buf.Size(), 0u);
    UNIT_ASSERT_EQUAL(b.Size(), sz);
}
}
;
