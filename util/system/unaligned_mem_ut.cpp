#include "unaligned_mem.h"

#include <library/cpp/testing/benchmark/bench.h>
#include <library/cpp/testing/unittest/registar.h>

#include <util/system/compiler.h>

#ifdef Y_HAVE_INT128
namespace {
    struct TUInt128 {
        bool operator==(const TUInt128& other) const {
            return x == other.x;
        }

        ui64 Low() const {
            return (ui64)x;
        }

        ui64 High() const {
            return (ui64)(x >> 64);
        }

        static TUInt128 Max() {
            return {~(__uint128_t)0};
        }

        __uint128_t x;
    };
} // namespace
#endif

Y_UNIT_TEST_SUITE(UnalignedMem) {
    Y_UNIT_TEST(TestReadWrite) {
        alignas(ui64) char buf[100];

        WriteUnaligned<ui16>(buf + 1, (ui16)1);
        WriteUnaligned<ui32>(buf + 1 + 2, (ui32)2);
        WriteUnaligned<ui64>(buf + 1 + 2 + 4, (ui64)3);

        UNIT_ASSERT_VALUES_EQUAL(ReadUnaligned<ui16>(buf + 1), 1);
        UNIT_ASSERT_VALUES_EQUAL(ReadUnaligned<ui32>(buf + 1 + 2), 2);
        UNIT_ASSERT_VALUES_EQUAL(ReadUnaligned<ui64>(buf + 1 + 2 + 4), 3);
    }

    Y_UNIT_TEST(TestReadWriteRuntime) {
        // Unlike the test above, this test avoids compile-time execution by a smart compiler.
        // It is required to catch the SIGSEGV in case compiler emits an alignment-sensitive instruction.

        alignas(ui64) static char buf[100] = {0}; // static is required for Clobber to work

        WriteUnaligned<ui16>(buf + 1, (ui16)1);
        WriteUnaligned<ui32>(buf + 1 + 2, (ui32)2);
        WriteUnaligned<ui64>(buf + 1 + 2 + 4, (ui64)3);
        NBench::Clobber();

        auto val1 = ReadUnaligned<ui16>(buf + 1);
        auto val2 = ReadUnaligned<ui32>(buf + 1 + 2);
        auto val3 = ReadUnaligned<ui64>(buf + 1 + 2 + 4);

        Y_DO_NOT_OPTIMIZE_AWAY(&val1);
        Y_DO_NOT_OPTIMIZE_AWAY(&val2);
        Y_DO_NOT_OPTIMIZE_AWAY(&val3);
        Y_DO_NOT_OPTIMIZE_AWAY(val1);
        Y_DO_NOT_OPTIMIZE_AWAY(val2);
        Y_DO_NOT_OPTIMIZE_AWAY(val3);

        UNIT_ASSERT_VALUES_EQUAL(val1, 1);
        UNIT_ASSERT_VALUES_EQUAL(val2, 2);
        UNIT_ASSERT_VALUES_EQUAL(val3, 3);
    }
#ifdef Y_HAVE_INT128
    Y_UNIT_TEST(TestReadWrite128) {
        alignas(TUInt128) char buf[100] = {0};

        WriteUnaligned<TUInt128>(buf + 1, TUInt128::Max());
        auto val = ReadUnaligned<TUInt128>(buf + 1);
        UNIT_ASSERT(val == TUInt128::Max());
    }
    Y_UNIT_TEST(TestReadWriteRuntime128) {
        // Unlike the test above, this test avoids compile-time execution by a smart compiler.
        // It is required to catch the SIGSEGV in case compiler emits an alignment-sensitive instruction.

        alignas(TUInt128) static char buf[100] = {0}; // static is required for Clobber to work

        WriteUnaligned<TUInt128>(buf + 1, TUInt128::Max());
        NBench::Clobber();

        auto val = ReadUnaligned<TUInt128>(buf + 1);
        Y_DO_NOT_OPTIMIZE_AWAY(&val);
        Y_DO_NOT_OPTIMIZE_AWAY(val.Low());
        Y_DO_NOT_OPTIMIZE_AWAY(val.High());

        UNIT_ASSERT(val == TUInt128::Max());
    }
#endif
} // Y_UNIT_TEST_SUITE(UnalignedMem)
