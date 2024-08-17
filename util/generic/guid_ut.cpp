#include <library/cpp/testing/unittest/registar.h>

#include "guid.h"

Y_UNIT_TEST_SUITE(TGuidTest) {
    // TODO - make real constructor
    static TGUID Construct(ui32 d1, ui32 d2, ui32 d3, ui32 d4) {
        TGUID ret;

        ret.dw[0] = d1;
        ret.dw[1] = d2;
        ret.dw[2] = d3;
        ret.dw[3] = d4;

        return ret;
    }

    struct TTest {
        TGUID G;
        TString S;
    };

    Y_UNIT_TEST(Test1) {
        for (size_t i = 0; i < 1000; ++i) {
            TGUID g;

            CreateGuid(&g);

            UNIT_ASSERT_EQUAL(g, GetGuid(GetGuidAsString(g)));
        }
    }

    Y_UNIT_TEST(Test2) {
        const TTest tests[] = {
            {Construct(1, 1, 1, 1), "1-1-1-1"},
            {Construct(0, 0, 0, 0), "0-0-0-0"},
            {TGUID(), "H-0-0-0"},
            {TGUID(), "0-H-0-0"},
            {TGUID(), "0-0-H-0"},
            {TGUID(), "0-0-0-H"},
            {Construct(0x8cf813d9U, 0xc098da90U, 0x7ef58954U, 0x636d04dU), "8cf813d9-c098da90-7ef58954-636d04d"},
            {Construct(0x8cf813d9U, 0xc098da90U, 0x7ef58954U, 0x636d04dU), "8CF813D9-C098DA90-7EF58954-636D04D"},
            {Construct(0x12345678U, 0x90abcdefU, 0xfedcba09U, 0x87654321U), "12345678-90abcdef-FEDCBA09-87654321"},
            {Construct(0x1, 0x2, 0xabcdef, 0x400), "01-002-00ABCDEF-000400"},
            {TGUID(), "-1-1-1"}, // empty parts
            {TGUID(), "--1-1-1"},
            {TGUID(), "1--1-1"},
            {TGUID(), "1-1"}, // unexpected end
            {TGUID(), "1-1-"},
            {TGUID(), "1-1-1"},
            {TGUID(), "1-1-1-"},
            {TGUID(), "1-1-1-1-"},
            {TGUID(), "1-1-1-1-1"},
            {TGUID(), "1+1-1-1"}, // bad char
            {TGUID(), "1-1:3-1-1"},
            {Construct(0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU), "FFFFFFFF-FFFFFFFF-FFFFFFFF-FFFFFFFF"}, // overflow
            {TGUID(), "FFFFFFFFA-FFFFFFFF-FFFFFFFF-FFFFFFFF"},
            {TGUID(), "100000000-0-0-0"},
            {Construct(1, 1, 1, 1), "0000001-0000000000000000000000000000000000000001-0001-00000001"},
            {Construct(0, 0, 0, 0), "000000000000-000000000000000000000000000000000000000-000-0"},
        };

        for (const auto& t : tests) {
            UNIT_ASSERT_EQUAL(t.G, GetGuid(t.S));
        }
    }

    Y_UNIT_TEST(Test3) {
        // if this test failed, please, fix buffer size in GetGuidAsString()
        TGUID max = Construct(Max<ui32>(), Max<ui32>(), Max<ui32>(), Max<ui32>());

        UNIT_ASSERT_EQUAL(GetGuidAsString(max).length(), 35);
    }

    Y_UNIT_TEST(Test4) {
        UNIT_ASSERT_VALUES_EQUAL(GetGuidAsString(Construct(1, 2, 3, 4)), "1-2-3-4");
        UNIT_ASSERT_VALUES_EQUAL(GetGuidAsString(Construct(1, 2, 0xFFFFFF, 4)), "1-2-ffffff-4");
        UNIT_ASSERT_VALUES_EQUAL(GetGuidAsString(Construct(0xFAFA, 2, 3, 4)), "fafa-2-3-4");
        UNIT_ASSERT_VALUES_EQUAL(GetGuidAsString(Construct(1, 0xADE, 3, 4)), "1-ade-3-4");
        UNIT_ASSERT_VALUES_EQUAL(GetGuidAsString(Construct(1, 2, 3, 0xDEAD)), "1-2-3-dead");
    }

    Y_UNIT_TEST(Test5) {
        const TTest tests[] = {
            {TGUID(), "1-1-1-1-1"},
            {TGUID(), "00000001-0001-0001-0001-00000000001-"},
            {Construct(0x10000001U, 0x10011001U, 0x10011001U, 0x10000001U), "10000001-1001-1001-1001-100110000001"},
            {Construct(0x550e8400U, 0xe29b41d4U, 0xa7164466U, 0x55440000U), "550e8400-e29b-41d4-a716-446655440000"},
            {Construct(0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU), "ffffffff-ffff-ffff-ffff-ffffffffffff"},
            {TGUID(), "ffffffff-ffffff-ff-ffff-ffffffffffff"},
            {TGUID(), "ffffffff-ffff-ffff-ff-ffffffffffffff"}};

        for (const auto& t : tests) {
            UNIT_ASSERT_EQUAL(t.G, GetUuid(t.S));
        }
    }

    Y_UNIT_TEST(DoubleConvert) {
        /**
         * test print and parsing RFC4122 GUID, which described in
         * https://en.wikipedia.org/wiki/Universally_unique_identifier
         * xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
         **/
        auto guid = TGUID::Create();
        auto printed = guid.AsUuidString();

        TGUID read;
        UNIT_ASSERT(GetUuid(printed, read));

        UNIT_ASSERT_VALUES_EQUAL(guid.dw[0], read.dw[0]);
        UNIT_ASSERT_VALUES_EQUAL(guid.dw[1], read.dw[1]);
        UNIT_ASSERT_VALUES_EQUAL(guid.dw[2], read.dw[2]);
        UNIT_ASSERT_VALUES_EQUAL(guid.dw[3], read.dw[3]);
    }

    Y_UNIT_TEST(OutputFormat) {
        TGUID guid = Construct(0x00005612U, 0x12000000U, 0x00000123U, 0x00000000U);

        UNIT_ASSERT_VALUES_EQUAL(guid.AsGuidString(), "5612-12000000-123-0");
        UNIT_ASSERT_VALUES_EQUAL(guid.AsUuidString(), "00005612-1200-0000-0000-012300000000");
    }

    Y_UNIT_TEST(TimeBased) {
        TString guid = TGUID::CreateTimebased().AsUuidString();
        UNIT_ASSERT(!guid.empty());
        UNIT_ASSERT_EQUAL(guid[14], '1');
    }
}
