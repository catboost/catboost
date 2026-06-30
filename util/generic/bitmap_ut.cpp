#include "bitmap.h"

#include <library/cpp/testing/unittest/registar.h>

#define INIT_BITMAP(bitmap, bits)                                \
    for (size_t i = 0; i < sizeof(bits) / sizeof(size_t); ++i) { \
        bitmap.Set(bits[i]);                                     \
    }

#define CHECK_BITMAP(bitmap, bits)                                      \
    {                                                                   \
        size_t cur = 0, end = sizeof(bits) / sizeof(size_t);            \
        for (size_t i = 0; i < bitmap.Size(); ++i) {                    \
            if (cur < end && bits[cur] == i) {                          \
                UNIT_ASSERT_EQUAL_C(bitmap.Get(i), true, "pos=" << i);  \
                ++cur;                                                  \
            } else {                                                    \
                UNIT_ASSERT_EQUAL_C(bitmap.Get(i), false, "pos=" << i); \
            }                                                           \
        }                                                               \
    }

#define CHECK_BITMAP_WITH_TAIL(bitmap, bits)                                \
    {                                                                       \
        size_t cur = 0, end = sizeof(bits) / sizeof(size_t);                \
        for (size_t i = 0; i < bitmap.Size(); ++i) {                        \
            if (cur < end) {                                                \
                if (bits[cur] == i) {                                       \
                    UNIT_ASSERT_EQUAL_C(bitmap.Get(i), true, "pos=" << i);  \
                    ++cur;                                                  \
                } else {                                                    \
                    UNIT_ASSERT_EQUAL_C(bitmap.Get(i), false, "pos=" << i); \
                }                                                           \
            } else {                                                        \
                UNIT_ASSERT_EQUAL_C(bitmap.Get(i), true, "pos=" << i);      \
            }                                                               \
        }                                                                   \
    }

Y_UNIT_TEST_SUITE(TBitMapTest) {
    Y_UNIT_TEST(TestBitMap) {
        TBitMap<101> bitmap;

        UNIT_ASSERT_EQUAL(bitmap.Size(), 101);
        UNIT_ASSERT_EQUAL(bitmap.Count(), 0);
        UNIT_ASSERT_EQUAL(bitmap.FirstNonZeroBit(), 101);

        size_t initBits[] = {0, 50, 100, 45};
        INIT_BITMAP(bitmap, initBits);
        bitmap.Reset(45);

        UNIT_ASSERT_EQUAL(bitmap.FirstNonZeroBit(), 0);
        size_t setBits[] = {0, 50, 100};
        CHECK_BITMAP(bitmap, setBits);

        for (size_t i = 0; i < bitmap.Size(); ++i) {
            UNIT_ASSERT_EQUAL(bitmap.Get(i), bitmap.Test(i));
        }

        UNIT_ASSERT_EQUAL(bitmap.Count(), 3);

        bitmap.Reset(0);

        UNIT_ASSERT_EQUAL(bitmap.FirstNonZeroBit(), 50);

        bitmap.Clear();

        UNIT_ASSERT_EQUAL(bitmap.Count(), 0);
        UNIT_ASSERT_EQUAL(bitmap.Empty(), true);
    }

    Y_UNIT_TEST(TestDynBitMap) {
        TDynBitMap bitmap;

        UNIT_ASSERT_EQUAL(bitmap.Size(), 64); // Initial capacity
        UNIT_ASSERT_EQUAL(bitmap.Count(), 0);
        UNIT_ASSERT_EQUAL(bitmap.FirstNonZeroBit(), 64);

        size_t initBits[] = {0, 50, 100, 45};
        INIT_BITMAP(bitmap, initBits);
        bitmap.Reset(45);

        UNIT_ASSERT_EQUAL(bitmap.Size(), 128);

        UNIT_ASSERT_EQUAL(bitmap.FirstNonZeroBit(), 0);
        size_t setBits[] = {0, 50, 100};
        CHECK_BITMAP(bitmap, setBits);

        for (size_t i = 0; i < bitmap.Size(); ++i) {
            UNIT_ASSERT_EQUAL(bitmap.Get(i), bitmap.Test(i));
        }

        UNIT_ASSERT_EQUAL(bitmap.Count(), 3);

        bitmap.Reset(0);

        UNIT_ASSERT_EQUAL(bitmap.FirstNonZeroBit(), 50);

        bitmap.Clear();

        UNIT_ASSERT_EQUAL(bitmap.Count(), 0);
        UNIT_ASSERT_EQUAL(bitmap.Empty(), true);
    }

    template <class TBitMapImpl>
    void TestAndImpl() {
        TBitMapImpl bitmap1;
        TBitMapImpl bitmap2;

        size_t initBits1[] = {10, 20, 50, 100};
        size_t initBits2[] = {10, 11, 22, 50};

        INIT_BITMAP(bitmap1, initBits1);
        INIT_BITMAP(bitmap2, initBits2);

        bitmap1.And(bitmap2);

        UNIT_ASSERT_EQUAL(bitmap1.Count(), 2);
        UNIT_ASSERT_EQUAL(bitmap2.Count(), 4);

        size_t resBits[] = {10, 50};

        CHECK_BITMAP(bitmap1, resBits);
        CHECK_BITMAP(bitmap2, initBits2);
    }

    Y_UNIT_TEST(TestAndFixed) {
        TestAndImpl<TBitMap<101>>();
    }

    Y_UNIT_TEST(TestAndDyn) {
        TestAndImpl<TDynBitMap>();
    }

    template <class TBitMapImpl>
    void TestOrImpl() {
        TBitMapImpl bitmap1;
        TBitMapImpl bitmap2;

        size_t initBits1[] = {0, 10, 11, 76};
        size_t initBits2[] = {1, 11, 22, 50};

        INIT_BITMAP(bitmap1, initBits1);
        INIT_BITMAP(bitmap2, initBits2);

        bitmap1.Or(bitmap2);

        UNIT_ASSERT_EQUAL(bitmap1.Count(), 7);
        UNIT_ASSERT_EQUAL(bitmap2.Count(), 4);

        size_t resBits[] = {0, 1, 10, 11, 22, 50, 76};

        CHECK_BITMAP(bitmap1, resBits);
        CHECK_BITMAP(bitmap2, initBits2);

        bitmap1.Clear();
        INIT_BITMAP(bitmap1, initBits1);

        UNIT_ASSERT_EQUAL(bitmap1 | (bitmap2 << 3), TBitMapImpl(bitmap1).Or(bitmap2, 3));
        UNIT_ASSERT_EQUAL(bitmap1 | (bitmap2 << 64), TBitMapImpl(bitmap1).Or(bitmap2, 64));
        UNIT_ASSERT_EQUAL(bitmap1 | (bitmap2 << 66), TBitMapImpl(bitmap1).Or(bitmap2, 66));

        UNIT_ASSERT_EQUAL(bitmap2 | (bitmap1 << 3), TBitMapImpl(bitmap2).Or(bitmap1, 3));
        UNIT_ASSERT_EQUAL(bitmap2 | (bitmap1 << 64), TBitMapImpl(bitmap2).Or(bitmap1, 64));
        UNIT_ASSERT_EQUAL(bitmap2 | (bitmap1 << 66), TBitMapImpl(bitmap2).Or(bitmap1, 66));
    }

    Y_UNIT_TEST(TestOrFixed) {
        TestOrImpl<TBitMap<145>>();
    }

    Y_UNIT_TEST(TestOrDyn) {
        TestOrImpl<TDynBitMap>();
    }

    Y_UNIT_TEST(TestCopy) {
        TBitMap<101> bitmap1;
        size_t initBits[] = {0, 10, 11, 76, 100};

        INIT_BITMAP(bitmap1, initBits);

        TDynBitMap bitmap2(bitmap1);
        CHECK_BITMAP(bitmap2, initBits);

        TBitMap<101> bitmap3(bitmap1);
        CHECK_BITMAP(bitmap3, initBits);

        TBitMap<127> bitmap4(bitmap1);
        CHECK_BITMAP(bitmap4, initBits);

        TDynBitMap bitmap5;
        bitmap5 = bitmap1;
        CHECK_BITMAP(bitmap5, initBits);

        TBitMap<101> bitmap6;
        bitmap6 = bitmap1;
        CHECK_BITMAP(bitmap6, initBits);

        TBitMap<127> bitmap7;
        bitmap7 = bitmap1;
        CHECK_BITMAP(bitmap7, initBits);

        TBitMap<101> bitmap8;
        DoSwap(bitmap8, bitmap6);
        CHECK_BITMAP(bitmap8, initBits);
        UNIT_ASSERT_EQUAL(bitmap6.Empty(), true);

        TDynBitMap bitmap9;
        DoSwap(bitmap9, bitmap5);
        CHECK_BITMAP(bitmap9, initBits);
        UNIT_ASSERT_EQUAL(bitmap5.Empty(), true);

        // 64->32
        TBitMap<160, ui32> bitmap10(bitmap1);
        CHECK_BITMAP(bitmap10, initBits);

        // 64->16
        TBitMap<160, ui16> bitmap11(bitmap1);
        CHECK_BITMAP(bitmap11, initBits);

        // 64->8
        TBitMap<160, ui8> bitmap12(bitmap1);
        CHECK_BITMAP(bitmap12, initBits);

        // 32->16
        TBitMap<160, ui16> bitmap13(bitmap10);
        CHECK_BITMAP(bitmap13, initBits);

        // 32->64
        TBitMap<160, ui64> bitmap14(bitmap10);
        CHECK_BITMAP(bitmap14, initBits);

        // 16->64
        TBitMap<160, ui64> bitmap15(bitmap11);
        CHECK_BITMAP(bitmap15, initBits);

        // 8->64
        TBitMap<160, ui64> bitmap16(bitmap12);
        CHECK_BITMAP(bitmap16, initBits);
    }

    Y_UNIT_TEST(TestOps) {
        TBitMap<16> bitmap1;
        TBitMap<12> bitmap2;
        size_t initBits1[] = {0, 3, 7, 11};
        size_t initBits2[] = {1, 3, 6, 7, 11};
        INIT_BITMAP(bitmap1, initBits1);
        INIT_BITMAP(bitmap2, initBits2);

        bitmap1.Or(3).And(bitmap2).Flip();

        size_t resBits[] = {0, 2, 4, 5, 6, 8, 9, 10, 12};
        CHECK_BITMAP_WITH_TAIL(bitmap1, resBits);

        TDynBitMap bitmap3;
        INIT_BITMAP(bitmap3, initBits1);

        bitmap3.Or(3).And(bitmap2).Flip();

        CHECK_BITMAP_WITH_TAIL(bitmap3, resBits);

        bitmap3.Clear();
        INIT_BITMAP(bitmap3, initBits1);

        TDynBitMap bitmap4 = ~((bitmap3 | 3) & bitmap2);
        CHECK_BITMAP_WITH_TAIL(bitmap4, resBits);

        TBitMap<128, ui32> expmap;
        expmap.Set(47);
        expmap.Set(90);
        ui64 tst1 = 0;
        ui32 tst2 = 0;
        ui16 tst3 = 0;
        expmap.Export(32, tst1);
        UNIT_ASSERT_EQUAL(tst1, (1 << 15) | (((ui64)1) << 58));
        expmap.Export(32, tst2);
        UNIT_ASSERT_EQUAL(tst2, (1 << 15));
        expmap.Export(32, tst3);
        UNIT_ASSERT_EQUAL(tst3, (1 << 15));

        expmap.Export(33, tst1);
        UNIT_ASSERT_EQUAL(tst1, (1 << 14) | (((ui64)1) << 57));
        expmap.Export(33, tst2);
        UNIT_ASSERT_EQUAL(tst2, (1 << 14));
        expmap.Export(33, tst3);
        UNIT_ASSERT_EQUAL(tst3, (1 << 14));
    }

    Y_UNIT_TEST(TestShiftFixed) {
        size_t initBits[] = {0, 3, 7, 11};

        TBitMap<128> bitmap1;

        INIT_BITMAP(bitmap1, initBits);
        bitmap1 <<= 62;
        size_t resBits1[] = {62, 65, 69, 73};
        CHECK_BITMAP(bitmap1, resBits1);
        bitmap1 >>= 62;
        CHECK_BITMAP(bitmap1, initBits);

        bitmap1.Clear();
        INIT_BITMAP(bitmap1, initBits);
        bitmap1 <<= 120;
        size_t resBits2[] = {120, 123, 127};
        CHECK_BITMAP(bitmap1, resBits2);
        bitmap1 >>= 120;
        size_t resBits3[] = {0, 3, 7};
        CHECK_BITMAP(bitmap1, resBits3);

        bitmap1.Clear();
        INIT_BITMAP(bitmap1, initBits);
        bitmap1 <<= 128;
        UNIT_ASSERT_EQUAL(bitmap1.Empty(), true);

        bitmap1.Clear();
        INIT_BITMAP(bitmap1, initBits);
        bitmap1 <<= 120;
        bitmap1 >>= 128;
        UNIT_ASSERT_EQUAL(bitmap1.Empty(), true);

        bitmap1.Clear();
        INIT_BITMAP(bitmap1, initBits);
        bitmap1 <<= 140;
        UNIT_ASSERT_EQUAL(bitmap1.Empty(), true);

        bitmap1.Clear();
        INIT_BITMAP(bitmap1, initBits);
        bitmap1 <<= 62;
        bitmap1 >>= 140;
        UNIT_ASSERT_EQUAL(bitmap1.Empty(), true);
    }

    Y_UNIT_TEST(TestShiftDyn) {
        size_t initBits[] = {0, 3, 7, 11};

        TDynBitMap bitmap1;

        INIT_BITMAP(bitmap1, initBits);
        bitmap1 <<= 62;
        size_t resBits1[] = {62, 65, 69, 73};
        CHECK_BITMAP(bitmap1, resBits1);
        bitmap1 >>= 62;
        CHECK_BITMAP(bitmap1, initBits);

        bitmap1.Clear();
        INIT_BITMAP(bitmap1, initBits);
        bitmap1 <<= 120;
        size_t resBits2[] = {120, 123, 127, 131};
        CHECK_BITMAP(bitmap1, resBits2);
        bitmap1 >>= 120;
        CHECK_BITMAP(bitmap1, initBits);

        bitmap1.Clear();
        INIT_BITMAP(bitmap1, initBits);
        bitmap1 <<= 128;
        size_t resBits3[] = {128, 131, 135, 139};
        CHECK_BITMAP(bitmap1, resBits3);

        bitmap1.Clear();
        INIT_BITMAP(bitmap1, initBits);
        bitmap1 <<= 120;
        bitmap1 >>= 128;
        size_t resBits4[] = {3};
        CHECK_BITMAP(bitmap1, resBits4);

        bitmap1.Clear();
        INIT_BITMAP(bitmap1, initBits);
        bitmap1 <<= 62;
        bitmap1 >>= 140;
        UNIT_ASSERT_EQUAL(bitmap1.Empty(), true);
    }

    // Test that we don't expand bitmap in LShift when high-order bits are zero
    Y_UNIT_TEST(TestShiftExpansion) {
        UNIT_ASSERT_EQUAL(TDynBitMap().LShift(1).Size(), 64);
        UNIT_ASSERT_EQUAL(TDynBitMap().LShift(65).Size(), 64);
        UNIT_ASSERT_EQUAL(TDynBitMap().LShift(128).Size(), 64);

        TDynBitMap bitmap;
        bitmap.Set(62).LShift(1);
        UNIT_ASSERT_EQUAL(bitmap, TDynBitMap().Set(63));
        UNIT_ASSERT_EQUAL(bitmap.Size(), 64);

        // Expand explicitly
        bitmap.Set(65);
        UNIT_ASSERT_EQUAL(bitmap.Size(), 128);

        bitmap.Clear().Set(0).LShift(1);
        UNIT_ASSERT_EQUAL(bitmap, TDynBitMap().Set(1));
        UNIT_ASSERT_EQUAL(bitmap.Size(), 128);

        bitmap.Clear().Set(63).LShift(1);
        UNIT_ASSERT_EQUAL(bitmap, TDynBitMap().Set(64));
        UNIT_ASSERT_EQUAL(bitmap.Size(), 128);

        bitmap.Clear().Set(63).LShift(64);
        UNIT_ASSERT_EQUAL(bitmap, TDynBitMap().Set(127));
        UNIT_ASSERT_EQUAL(bitmap.Size(), 128);

        bitmap.Clear().Set(62).LShift(129);
        UNIT_ASSERT_EQUAL(bitmap, TDynBitMap().Set(191));
        UNIT_ASSERT_EQUAL(bitmap.Size(), 256);
    }

    Y_UNIT_TEST(TestFixedSanity) {
        // test extra-bit cleanup
        UNIT_ASSERT_EQUAL(TBitMap<33>().Set(0).LShift(34).RShift(34).Empty(), true);
        UNIT_ASSERT_EQUAL(TBitMap<88>().Set(0).Set(1).Set(2).LShift(90).RShift(90).Empty(), true);
        UNIT_ASSERT_EQUAL(TBitMap<88>().Flip().RShift(88).Empty(), true);
        UNIT_ASSERT_EQUAL(TBitMap<64>().Flip().LShift(2).RShift(2).Count(), 62);
        UNIT_ASSERT_EQUAL(TBitMap<67>().Flip().LShift(2).RShift(2).Count(), 65);
        UNIT_ASSERT_EQUAL(TBitMap<128>().Flip().LShift(2).RShift(2).Count(), 126);
        UNIT_ASSERT_EQUAL(TBitMap<130>().Flip().LShift(2).RShift(2).Count(), 128);
        UNIT_ASSERT_EQUAL(TBitMap<130>(TDynBitMap().Set(131)).Empty(), true);
        UNIT_ASSERT_EQUAL(TBitMap<33>().Or(TBitMap<40>().Set(39)).Empty(), true);
        UNIT_ASSERT_EQUAL(TBitMap<33>().Xor(TBitMap<40>().Set(39)).Empty(), true);
    }

    Y_UNIT_TEST(TestIterate) {
        TDynBitMap bitmap1;
        TDynBitMap bitmap2;

        size_t initBits1[] = {0, 3, 7, 8, 11, 33, 34, 35, 36, 62, 63, 100, 127};
        INIT_BITMAP(bitmap1, initBits1);
        for (size_t i = bitmap1.FirstNonZeroBit(); i != bitmap1.Size(); i = bitmap1.NextNonZeroBit(i)) {
            bitmap2.Set(i);
        }
        CHECK_BITMAP(bitmap2, initBits1);
        UNIT_ASSERT_EQUAL(bitmap1, bitmap2);

        size_t initBits2[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 33, 34, 35, 36, 62};
        bitmap1.Clear();
        bitmap2.Clear();
        INIT_BITMAP(bitmap1, initBits2);
        for (size_t i = bitmap1.FirstNonZeroBit(); i != bitmap1.Size(); i = bitmap1.NextNonZeroBit(i)) {
            bitmap2.Set(i);
        }
        CHECK_BITMAP(bitmap2, initBits2);
        UNIT_ASSERT_EQUAL(bitmap1, bitmap2);

        UNIT_ASSERT_EQUAL(bitmap1.NextNonZeroBit(63), bitmap1.Size());
        UNIT_ASSERT_EQUAL(bitmap1.NextNonZeroBit(64), bitmap1.Size());
        UNIT_ASSERT_EQUAL(bitmap1.NextNonZeroBit(65), bitmap1.Size());
        UNIT_ASSERT_EQUAL(bitmap1.NextNonZeroBit(127), bitmap1.Size());
        UNIT_ASSERT_EQUAL(bitmap1.NextNonZeroBit(533), bitmap1.Size());

        TBitMap<128, ui8> bitmap3;
        bitmap1.Clear();
        INIT_BITMAP(bitmap1, initBits1);
        for (size_t i = bitmap1.FirstNonZeroBit(); i != bitmap1.Size(); i = bitmap1.NextNonZeroBit(i)) {
            bitmap3.Set(i);
        }
        CHECK_BITMAP(bitmap3, initBits1);
        UNIT_ASSERT_EQUAL(bitmap3, bitmap1);

        TBitMap<18> bitmap4;
        bitmap4.Set(15);
        UNIT_ASSERT_EQUAL(bitmap4.NextNonZeroBit(0), 15);
        UNIT_ASSERT_EQUAL(bitmap4.NextNonZeroBit(15), bitmap4.Size());
        UNIT_ASSERT_EQUAL(bitmap4.NextNonZeroBit(63), bitmap4.Size());
        UNIT_ASSERT_EQUAL(bitmap4.NextNonZeroBit(64), bitmap4.Size());
        UNIT_ASSERT_EQUAL(bitmap4.NextNonZeroBit(65), bitmap4.Size());
        UNIT_ASSERT_EQUAL(bitmap4.NextNonZeroBit(127), bitmap4.Size());
        UNIT_ASSERT_EQUAL(bitmap4.NextNonZeroBit(533), bitmap4.Size());

        bitmap4.Clear().Flip();
        UNIT_ASSERT_EQUAL(bitmap4.NextNonZeroBit(0), 1);
        UNIT_ASSERT_EQUAL(bitmap4.NextNonZeroBit(15), 16);
        UNIT_ASSERT_EQUAL(bitmap4.NextNonZeroBit(17), bitmap4.Size());
        UNIT_ASSERT_EQUAL(bitmap4.NextNonZeroBit(18), bitmap4.Size());
        UNIT_ASSERT_EQUAL(bitmap4.NextNonZeroBit(63), bitmap4.Size());
        UNIT_ASSERT_EQUAL(bitmap4.NextNonZeroBit(64), bitmap4.Size());
        UNIT_ASSERT_EQUAL(bitmap4.NextNonZeroBit(65), bitmap4.Size());
        UNIT_ASSERT_EQUAL(bitmap4.NextNonZeroBit(127), bitmap4.Size());
        UNIT_ASSERT_EQUAL(bitmap4.NextNonZeroBit(533), bitmap4.Size());
    }

    Y_UNIT_TEST(TestHashFixed) {
        TBitMap<32, ui8> bitmap32;
        TBitMap<32, ui8> bitmap322;
        TBitMap<64, ui8> bitmap64;

        bitmap32.Clear();
        bitmap322.Clear();
        UNIT_ASSERT_EQUAL(bitmap32.Hash(), bitmap322.Hash());
        bitmap32.Set(0);
        UNIT_ASSERT_UNEQUAL(bitmap32.Hash(), bitmap322.Hash());
        bitmap322.Set(0);
        UNIT_ASSERT_EQUAL(bitmap32.Hash(), bitmap322.Hash());
        bitmap32.Set(8).Set(31);
        UNIT_ASSERT_UNEQUAL(bitmap32.Hash(), bitmap322.Hash());
        bitmap322.Set(8).Set(31);
        UNIT_ASSERT_EQUAL(bitmap32.Hash(), bitmap322.Hash());

        bitmap32.Clear();
        bitmap64.Clear();
        UNIT_ASSERT_EQUAL(bitmap32.Hash(), bitmap64.Hash());
        bitmap32.Set(0);
        UNIT_ASSERT_UNEQUAL(bitmap32.Hash(), bitmap64.Hash());
        bitmap64.Set(0);
        UNIT_ASSERT_EQUAL(bitmap32.Hash(), bitmap64.Hash());
        bitmap32.Set(8).Set(31);
        UNIT_ASSERT_UNEQUAL(bitmap32.Hash(), bitmap64.Hash());
        bitmap64.Set(8).Set(31);
        UNIT_ASSERT_EQUAL(bitmap32.Hash(), bitmap64.Hash());
        bitmap64.Set(32);
        UNIT_ASSERT_UNEQUAL(bitmap32.Hash(), bitmap64.Hash());
    }

    Y_UNIT_TEST(TestHashDynamic) {
        TDynBitMap bitmap1;
        TDynBitMap bitmap2;

        bitmap1.Clear();
        bitmap2.Clear();
        UNIT_ASSERT_EQUAL(bitmap1.Hash(), bitmap2.Hash());
        bitmap1.Set(0);
        UNIT_ASSERT_UNEQUAL(bitmap1.Hash(), bitmap2.Hash());
        bitmap2.Set(0);
        UNIT_ASSERT_EQUAL(bitmap1.Hash(), bitmap2.Hash());
        bitmap1.Set(8).Set(31);
        UNIT_ASSERT_UNEQUAL(bitmap1.Hash(), bitmap2.Hash());
        bitmap2.Set(8).Set(31);
        UNIT_ASSERT_EQUAL(bitmap1.Hash(), bitmap2.Hash());
        bitmap1.Set(64);
        UNIT_ASSERT_UNEQUAL(bitmap1.Hash(), bitmap2.Hash());
        bitmap2.Set(64);
        UNIT_ASSERT_EQUAL(bitmap1.Hash(), bitmap2.Hash());
    }

    Y_UNIT_TEST(TestHashMixed) {
        static_assert((std::is_same<TDynBitMap::TChunk, ui64>::value), "expect (TSameType<TDynBitMap::TChunk, ui64>::Result)");

        TBitMap<sizeof(ui64) * 16, ui64> bitmapFixed;
        TDynBitMap bitmapDynamic;

        bitmapFixed.Clear();
        bitmapDynamic.Clear();
        UNIT_ASSERT_EQUAL(bitmapFixed.Hash(), bitmapDynamic.Hash());
        bitmapFixed.Set(0);
        UNIT_ASSERT_UNEQUAL(bitmapFixed.Hash(), bitmapDynamic.Hash());
        bitmapDynamic.Set(0);
        UNIT_ASSERT_EQUAL(bitmapFixed.Hash(), bitmapDynamic.Hash());
        bitmapFixed.Set(8).Set(127);
        UNIT_ASSERT_UNEQUAL(bitmapFixed.Hash(), bitmapDynamic.Hash());
        bitmapDynamic.Set(8).Set(127);
        UNIT_ASSERT_EQUAL(bitmapFixed.Hash(), bitmapDynamic.Hash());
        bitmapDynamic.Set(128);
        UNIT_ASSERT_UNEQUAL(bitmapFixed.Hash(), bitmapDynamic.Hash());
    }

    Y_UNIT_TEST(TestSetResetRange) {
        // Single chunk
        using TBitMap1Chunk = TBitMap<64>;
        UNIT_ASSERT_EQUAL(TBitMap1Chunk().Flip().Reset(10, 50), TBitMap1Chunk().Set(0, 10).Set(50, 64));
        UNIT_ASSERT_EQUAL(TBitMap1Chunk().Flip().Reset(0, 10), TBitMap1Chunk().Set(10, 64));
        UNIT_ASSERT_EQUAL(TBitMap1Chunk().Flip().Reset(50, 64), TBitMap1Chunk().Set(0, 50));
        UNIT_ASSERT_EQUAL(TBitMap1Chunk().Flip().Reset(0, 10).Reset(50, 64), TBitMap1Chunk().Set(10, 50));

        // Two chunks
        using TBitMap2Chunks = TBitMap<64, ui32>;
        UNIT_ASSERT_EQUAL(TBitMap2Chunks().Flip().Reset(10, 50), TBitMap2Chunks().Set(0, 10).Set(50, 64));
        UNIT_ASSERT_EQUAL(TBitMap2Chunks().Flip().Reset(0, 10), TBitMap2Chunks().Set(10, 64));
        UNIT_ASSERT_EQUAL(TBitMap2Chunks().Flip().Reset(50, 64), TBitMap2Chunks().Set(0, 50));
        UNIT_ASSERT_EQUAL(TBitMap2Chunks().Flip().Reset(0, 10).Reset(50, 64), TBitMap2Chunks().Set(10, 50));

        // Many chunks
        using TBitMap4Chunks = TBitMap<64, ui16>;
        UNIT_ASSERT_EQUAL(TBitMap4Chunks().Flip().Reset(10, 50), TBitMap4Chunks().Set(0, 10).Set(50, 64));
        UNIT_ASSERT_EQUAL(TBitMap4Chunks().Flip().Reset(0, 10), TBitMap4Chunks().Set(10, 64));
        UNIT_ASSERT_EQUAL(TBitMap4Chunks().Flip().Reset(50, 64), TBitMap4Chunks().Set(0, 50));
        UNIT_ASSERT_EQUAL(TBitMap4Chunks().Flip().Reset(0, 10).Reset(50, 64), TBitMap4Chunks().Set(10, 50));
    }

    Y_UNIT_TEST(TestSetRangeDyn) {
        for (size_t start = 0; start < 192; ++start) {
            for (size_t end = start; end < 192; ++end) {
                TDynBitMap bm;
                bm.Reserve(192);
                bm.Set(start, end);
                for (size_t k = 0; k < 192; ++k) {
                    UNIT_ASSERT_VALUES_EQUAL(bm.Get(k), k >= start && k < end ? 1 : 0);
                }
            }
        }
    }

    Y_UNIT_TEST(TestResetLargeRangeDyn) {
        TDynBitMap bm;
        bm.Set(0);
        bm.Reset(1, 2048);
        bm.Set(2048);
        for (size_t k = 0; k <= 2048; ++k) {
            UNIT_ASSERT_VALUES_EQUAL(bm.Get(k), k >= 1 && k < 2048 ? 0 : 1);
        }
    }
} // Y_UNIT_TEST_SUITE(TBitMapTest)
