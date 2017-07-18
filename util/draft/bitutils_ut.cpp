#include "bitutils.h"

#include <library/unittest/registar.h>

#include <util/string/util.h>

class TBitUtilsTest: public TTestBase {
    UNIT_TEST_SUITE(TBitUtilsTest);
    UNIT_TEST(TestK)
    UNIT_TEST_SUITE_END();

private:
    template <ui64 bits>
    void DoTestK() {
        using namespace NBitUtils;
        UNIT_ASSERT_VALUES_EQUAL_C(MaskK<bits>(), MaskLowerBits(bits), bits);
        UNIT_ASSERT_VALUES_EQUAL_C(FlagK<bits>(), Flag(bits), bits);

        for (ui32 i = 0; i < 65; ++i) {
            UNIT_ASSERT_VALUES_EQUAL_C(MaskK<bits>(i), MaskLowerBits(bits, i), bits);
        }
    }

    void TestK() {
        DoTestK<0>();
        DoTestK<1>();
        DoTestK<2>();
        DoTestK<3>();
        DoTestK<4>();
        DoTestK<5>();
        DoTestK<6>();
        DoTestK<7>();
        DoTestK<8>();
        DoTestK<9>();
        DoTestK<10>();
        DoTestK<12>();
        DoTestK<13>();
        DoTestK<14>();
        DoTestK<15>();
        DoTestK<16>();
        DoTestK<17>();
        DoTestK<18>();
        DoTestK<19>();
        DoTestK<20>();
        DoTestK<21>();
        DoTestK<22>();
        DoTestK<23>();
        DoTestK<24>();
        DoTestK<25>();
        DoTestK<26>();
        DoTestK<27>();
        DoTestK<28>();
        DoTestK<29>();
        DoTestK<30>();
        DoTestK<31>();
        DoTestK<32>();
        DoTestK<33>();
        DoTestK<34>();
        DoTestK<35>();
        DoTestK<36>();
        DoTestK<37>();
        DoTestK<38>();
        DoTestK<39>();
        DoTestK<40>();
        DoTestK<41>();
        DoTestK<42>();
        DoTestK<43>();
        DoTestK<44>();
        DoTestK<45>();
        DoTestK<46>();
        DoTestK<47>();
        DoTestK<48>();
        DoTestK<49>();
        DoTestK<50>();
        DoTestK<51>();
        DoTestK<52>();
        DoTestK<53>();
        DoTestK<54>();
        DoTestK<55>();
        DoTestK<56>();
        DoTestK<57>();
        DoTestK<58>();
        DoTestK<59>();
        DoTestK<60>();
        DoTestK<61>();
        DoTestK<62>();
        DoTestK<63>();
        DoTestK<64>();
    }
};

UNIT_TEST_SUITE_REGISTRATION(TBitUtilsTest);
