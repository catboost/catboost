#include "memory.h"

#include <library/cpp/testing/unittest/registar.h>

#pragma pack(1)
struct Y_PACKED TSampleStruct1 {
    ui8 A;
    ui8 B;
};

#pragma pack(1)
struct Y_PACKED TSampleStruct2 {
    ui8 A;
    ui16 B;
    i32 C;
};

#pragma pack(1)
struct Y_PACKED TSampleStruct3 {
    TSampleStruct2 A;
    ui64 B;
};

#pragma pack()

Y_UNIT_TEST_SUITE(TUtilDraftMemoryTest) {
    Y_UNIT_TEST(IsZeroTest) {
        ui8 a1 = 0;
        UNIT_ASSERT(IsZero(a1));
        a1 = 0xF0;
        UNIT_ASSERT(!IsZero(a1));

        i32 a2 = -1;
        UNIT_ASSERT(!IsZero(a2));
        a2 = 0;
        UNIT_ASSERT(IsZero(a2));

        double a3 = 0.0;
        UNIT_ASSERT(IsZero(a3));
        a3 = 1.e-13;
        UNIT_ASSERT(!IsZero(a3));

        TSampleStruct1 ss1;
        ss1.A = 0;
        ss1.B = 0;
        UNIT_ASSERT(IsZero(ss1));
        ss1.A = 0;
        ss1.B = 12;
        UNIT_ASSERT(!IsZero(ss1));

        TSampleStruct2 ss2;
        ss2.A = 0;
        ss2.B = 100;
        ss2.C = 0;
        UNIT_ASSERT(!IsZero(ss2));
        ss2.B = 0;
        UNIT_ASSERT(IsZero(ss2));

        TSampleStruct3 ss3;
        ss3.A = ss2;
        ss3.B = 0;
        UNIT_ASSERT(IsZero(ss3));
        ss3.B = 0x030;
        UNIT_ASSERT(!IsZero(ss3));
        ss3.B = 0;
        ss3.A.C = -789;
        UNIT_ASSERT(!IsZero(ss3));
    }
} // Y_UNIT_TEST_SUITE(TUtilDraftMemoryTest)
