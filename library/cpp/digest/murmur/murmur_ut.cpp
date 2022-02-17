#include "murmur.h"

#include <library/cpp/testing/unittest/registar.h>

class TMurmurHashTest: public TTestBase {
private:
    UNIT_TEST_SUITE(TMurmurHashTest);
    UNIT_TEST(TestHash2A32)
    UNIT_TEST(TestHash2A64)
    UNIT_TEST(TestUnaligned)
    UNIT_TEST_SUITE_END();

private:
    inline void TestHash2A32() {
        ui8 buf[256];

        for (size_t i = 0; i < 256; ++i) {
            buf[i] = i;
        }

        Test2A<ui32>(buf, 256, 0, 97, 178525084UL);
        Test2A<ui32>(buf, 256, 128, 193, 178525084UL);
        Test2A<ui32>(buf, 255, 0, 97, 2459858906UL);
        Test2A<ui32>(buf, 255, 128, 193, 2459858906UL);
    }

    inline void TestHash2A64() {
        ui8 buf[256];

        for (size_t i = 0; i < 256; ++i) {
            buf[i] = i;
        }

        Test2A<ui64>(buf, 256, 0, 97, ULL(15099340606808450747));
        Test2A<ui64>(buf, 256, 128, 193, ULL(15099340606808450747));
        Test2A<ui64>(buf, 255, 0, 97, ULL(8331973280124075880));
        Test2A<ui64>(buf, 255, 128, 193, ULL(8331973280124075880));
    }

    inline void TestUnaligned() {
        ui8 buf[257];
        for (size_t i = 0; i < 256; ++i) {
            buf[i + 1] = i;
        }
        Test2A<ui64>(buf + 1, 256, 0, 97, ULL(15099340606808450747));
        Test2A<ui64>(buf + 1, 256, 128, 193, ULL(15099340606808450747));
        Test2A<ui64>(buf + 1, 255, 0, 97, ULL(8331973280124075880));
        Test2A<ui64>(buf + 1, 255, 128, 193, ULL(8331973280124075880));
    }

private:
    template <class T>
    static inline void Test2A(const ui8* data, size_t len, size_t split1, size_t split2, T expected) {
        const T value1 = TMurmurHash2A<T>().Update(data, split1).Update(data + split1, len - split1).Value();
        const T value2 = TMurmurHash2A<T>().Update(data, split2).Update(data + split2, len - split2).Value();
        UNIT_ASSERT_VALUES_EQUAL(value1, value2);
        UNIT_ASSERT_VALUES_EQUAL(value1, expected);
    }
};

UNIT_TEST_SUITE_REGISTRATION(TMurmurHashTest);
