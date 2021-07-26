
#include "murmur.h"

#include <library/cpp/testing/unittest/registar.h>

class TMurmurHashTest: public TTestBase {
    UNIT_TEST_SUITE(TMurmurHashTest);
    UNIT_TEST(TestHash32)
    UNIT_TEST(TestUnalignedHash32)
    UNIT_TEST(TestHash64)
    UNIT_TEST(TestUnalignedHash64)
    UNIT_TEST(TestWrapperBiggerTypes)
    UNIT_TEST_SUITE_END();

private:
    inline void TestHash32() {
        ui8 buf[256];

        for (size_t i = 0; i < 256; ++i) {
            buf[i] = i;
        }

        Test<ui32>(buf, 256, 2373126550UL);
        TestWrapper<ui8, ui32>({buf, buf + 256}, 2373126550UL);
        Test<ui32>(buf, 255, 3301607533UL);
        Test<ui32>(buf, 254, 2547410121UL);
        Test<ui32>(buf, 253, 80030810UL);
    }

    inline void TestUnalignedHash32() {
        ui8 buf[257];
        ui8* unalignedBuf = buf + 1;

        for (size_t i = 0; i < 256; ++i) {
            unalignedBuf[i] = i;
        }

        Test<ui32>(unalignedBuf, 256, 2373126550UL);
    }

    inline void TestHash64() {
        ui8 buf[256];

        for (size_t i = 0; i < 256; ++i) {
            buf[i] = i;
        }

        Test<ui64>(buf, 256, ULL(12604435678857905857));
        TestWrapper<ui8, ui64>({buf, buf + 256}, ULL(12604435678857905857));
        Test<ui64>(buf, 255, ULL(1708835094528446095));
        Test<ui64>(buf, 254, ULL(5077937678736514994));
        Test<ui64>(buf, 253, ULL(11553864555081396353));
    }

    inline void TestUnalignedHash64() {
        ui8 buf[257];
        ui8* unalignedBuf = buf + 1;

        for (size_t i = 0; i < 256; ++i) {
            unalignedBuf[i] = i;
        }

        Test<ui64>(unalignedBuf, 256, ULL(12604435678857905857));
    }

    inline void TestWrapperBiggerTypes() {
        ui32 buf[] = {24, 42};
        TestWrapper<ui32, ui32>({buf, buf + 2}, MurmurHash<ui32>(buf, sizeof(ui32) * 2));
        TestWrapper<ui32, ui64>({buf, buf + 2}, MurmurHash<ui64>(buf, sizeof(ui32) * 2));
    }

private:
    template <class T>
    inline void Test(const void* data, size_t len, T expected) {
        UNIT_ASSERT_STRINGS_EQUAL(ToString(MurmurHash<T>(data, len)), ToString(expected));
    }

    template <class E, class T>
    inline void TestWrapper(const TArrayRef<E>& array, T expected) {
        auto val = TMurmurHash<T>()(array);
        UNIT_ASSERT_EQUAL(val, expected);
    }
};

UNIT_TEST_SUITE_REGISTRATION(TMurmurHashTest);
