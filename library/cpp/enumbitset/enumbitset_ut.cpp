#include "enumbitset.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/stream/str.h>
#include <util/stream/file.h>

enum ETestEnum {
    TE_FIRST = 0,
    TE_1,
    TE_2,
    TE_3,
    TE_4,
    TE_MIDDLE = TE_4 + 64,
    TE_5,
    TE_6,
    TE_7,
    TE_MAX,

    TE_OVERFLOW, // to test overflow
};
using TTestBitSet = TEnumBitSet<ETestEnum, TE_FIRST, TE_MAX>;

namespace {
    enum class EStorageTest : int {};
    struct TThrowingEnumConversion {
        operator ETestEnum() const {
            ythrow yexception() << "Enum conversion failed";
        }
    };

    template <class T, class = void>
    struct THasToMask: std::false_type {};

    template <class T>
    struct THasToMask<T, std::void_t<decltype(std::declval<const T&>().ToMask())>>: std::true_type {};

    template <int Begin, int Size, typename TExpectedChunk>
    void CheckCompactStorage(TStringBuf fullHex) {
        using TSet = TEnumBitSet<EStorageTest, Begin, Begin + Size>;
        static_assert(std::is_same_v<typename TSet::TChunk, TExpectedChunk>);
        static_assert(std::is_same_v<decltype(TSet{}.Low()), ui64>);
        static_assert(THasToMask<TSet>::value == (Size <= 64));
        static_assert(sizeof(TSet) == sizeof(TExpectedChunk) * ((Size + 8 * sizeof(TExpectedChunk) - 1) / (8 * sizeof(TExpectedChunk))));

        const auto first = static_cast<EStorageTest>(Begin);
        const auto last = static_cast<EStorageTest>(Begin + Size - 1);
        TSet empty;
        UNIT_ASSERT(empty.Empty());
        UNIT_ASSERT_VALUES_EQUAL(empty.Low(), 0);
        UNIT_ASSERT_VALUES_EQUAL(empty.ToString(), "0");

        const TSet firstOnly(first);
        const TSet lastOnly(last);
        UNIT_ASSERT(TSet(TBitMap<Size>(1)) == firstOnly);
        auto endpoints = firstOnly | lastOnly;
        UNIT_ASSERT((firstOnly | last) == endpoints);
        UNIT_ASSERT((firstOnly ^ last) == (Size == 1 ? empty : endpoints));
        auto edited = firstOnly;
        UNIT_ASSERT(&(edited |= last) == &edited);
        UNIT_ASSERT(edited == endpoints);
        edited |= last;
        UNIT_ASSERT(edited == endpoints);
        UNIT_ASSERT(&(edited ^= last) == &edited);
        UNIT_ASSERT(edited == (Size == 1 ? empty : firstOnly));
        edited ^= last;
        UNIT_ASSERT(edited == endpoints);
        UNIT_ASSERT((endpoints & firstOnly) == firstOnly);
        UNIT_ASSERT((endpoints & ~firstOnly) == (Size == 1 ? empty : lastOnly));
        endpoints.Set(last);
        UNIT_ASSERT_VALUES_EQUAL(endpoints.Count(), Size == 1 ? 1 : 2);
        endpoints.Reset(first);
        UNIT_ASSERT(!endpoints.Test(first));
        endpoints.Flip(first);
        UNIT_ASSERT(endpoints.Test(first));
        UNIT_ASSERT_VALUES_EQUAL(lastOnly.Low(), Size <= 64 ? ui64{1} << ((Size - 1) % 64) : 0);

        const auto full = ~empty;
        UNIT_ASSERT_VALUES_EQUAL(full.Count(), Size);
        UNIT_ASSERT_VALUES_EQUAL(full.Low(), ~ui64{0} >> (64 - Min(Size, 64)));
        if constexpr (Size <= 64) {
            static_assert(std::is_same_v<decltype(full.ToMask()), TExpectedChunk>);
            UNIT_ASSERT_VALUES_EQUAL(empty.ToMask(), 0);
            UNIT_ASSERT_VALUES_EQUAL(lastOnly.ToMask(), lastOnly.Low());
            UNIT_ASSERT_VALUES_EQUAL(full.ToMask(), full.Low());
        }
        UNIT_ASSERT_VALUES_EQUAL(full.ToString(), fullHex);
        int next = Begin;
        for (const auto value : full) {
            UNIT_ASSERT_VALUES_EQUAL(static_cast<int>(value), next++);
        }
        UNIT_ASSERT_VALUES_EQUAL(next, Begin + Size);

        TSet parsed;
        parsed.FromString(full.ToString());
        UNIT_ASSERT(parsed == full);
        const TString paddedHex = TString(((fullHex.size() + 15) / 16) * 16 - fullHex.size(), '0') + fullHex;
        parsed.FromString(paddedHex);
        UNIT_ASSERT(parsed == full);
        parsed.FromString("0");
        UNIT_ASSERT(parsed.Empty());
        UNIT_ASSERT(!parsed.TryFromString(TString(fullHex) + "0"));
        UNIT_ASSERT(parsed.Empty());

        // Keep the wire format independent of the in-memory chunk width.
        TStringStream serialized;
        full.Save(&serialized);
        TStringStream expected;
        ::Save(&expected, ui32{Size});
        for (int value = Begin; value < Begin + Size; ++value) {
            ::Save(&expected, static_cast<ui32>(value));
        }
        UNIT_ASSERT_VALUES_EQUAL(serialized.Str(), expected.Str());
        TSet restored;
        restored.Load(&serialized);
        UNIT_ASSERT(restored == full);
    }
}

Y_UNIT_TEST_SUITE(TEnumBitSetTest) {
    Y_UNIT_TEST(TestCompactStorage) {
        CheckCompactStorage<0, 1, ui8>("1");
        CheckCompactStorage<0, 7, ui8>("7F");
        CheckCompactStorage<0, 8, ui8>("FF");
        CheckCompactStorage<0, 9, ui16>("1FF");
        CheckCompactStorage<0, 16, ui16>("FFFF");
        CheckCompactStorage<0, 17, ui32>("1FFFF");
        CheckCompactStorage<0, 32, ui32>("FFFFFFFF");
        CheckCompactStorage<0, 33, ui64>("1FFFFFFFF");
        CheckCompactStorage<0, 64, ui64>("FFFFFFFFFFFFFFFF");
        CheckCompactStorage<0, 65, ui64>("1FFFFFFFFFFFFFFFF");
        CheckCompactStorage<0, 128, ui64>("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF");
        CheckCompactStorage<100, 8, ui8>("FF");
        CheckCompactStorage<-10, 9, ui16>("1FF");
    }

    Y_UNIT_TEST(TestSafeCompactStorage) {
        using TSmallSet = TSfEnumBitSet<ETestEnum, TE_FIRST, 8>;
        static_assert(sizeof(TSmallSet) == sizeof(ui8));
        TSmallSet set(~ui64{0});
        UNIT_ASSERT_VALUES_EQUAL(set.Low(), 0xFF);
        UNIT_ASSERT_VALUES_EQUAL(set.Count(), 8);
        set.SafeReset(TE_1);
        UNIT_ASSERT(!set.SafeTest(TE_1));
        set.SafeSet(TE_OVERFLOW);
        UNIT_ASSERT_VALUES_EQUAL(set.Count(), 7);
        UNIT_ASSERT(TSmallSet::GetFromString("00000000000000FD") == set);

        const char framedHex[] = {'x', 'F', 'D', 'y'};
        const TStringBuf hex(framedHex + 1, 2);
        UNIT_ASSERT(TSmallSet::GetFromString(hex) == set);
        UNIT_ASSERT(TSmallSet::TryGetFromString(hex) == set);
        const TStringBuf invalidHex(framedHex + 1, 3);
        UNIT_ASSERT_EXCEPTION(TSmallSet::GetFromString(invalidHex), yexception);
        UNIT_ASSERT(TSmallSet::TryGetFromString(invalidHex).Empty());

        using TScopedSet = TSfEnumBitSet<EStorageTest, static_cast<EStorageTest>(-1), 7>;
        const auto first = static_cast<EStorageTest>(-1);
        const auto last = static_cast<EStorageTest>(6);
        TScopedSet scoped(first, last, static_cast<EStorageTest>(7));
        UNIT_ASSERT_VALUES_EQUAL(scoped.Count(), 2);
        UNIT_ASSERT(scoped.Test(first));
        UNIT_ASSERT(scoped.Test(last));
        UNIT_ASSERT(TScopedSet(ui64{0x81}) == scoped);
    }

    Y_UNIT_TEST(TestMainFunctions) {
        auto ebs = TTestBitSet(TE_FIRST, TE_MIDDLE);

        UNIT_ASSERT(ebs.SafeTest(TE_FIRST));
        UNIT_ASSERT(ebs.SafeTest(TE_MIDDLE));

        ebs.SafeSet(TE_5);
        UNIT_ASSERT(ebs.SafeTest(TE_5));
        UNIT_ASSERT(!ebs.SafeTest(TE_7));

        ebs.SafeSet(TE_OVERFLOW);
        UNIT_ASSERT(!ebs.SafeTest(TE_OVERFLOW));
    }

    Y_UNIT_TEST(TestValidationBounds) {
        using TNegativeBegin = TEnumBitSet<EStorageTest, -1, 4>;
        UNIT_ASSERT(TNegativeBegin::IsValid(static_cast<EStorageTest>(-1)));
        UNIT_ASSERT(!TNegativeBegin::IsValid(static_cast<EStorageTest>(-2)));
        UNIT_ASSERT(TNegativeBegin::IsValid(static_cast<EStorageTest>(3)));
        UNIT_ASSERT(!TNegativeBegin::IsValid(static_cast<EStorageTest>(4)));

        using TNegativeEnd = TEnumBitSet<EStorageTest, -4, -1>;
        UNIT_ASSERT(TNegativeEnd::IsValid(static_cast<EStorageTest>(-4)));
        UNIT_ASSERT(!TNegativeEnd::IsValid(static_cast<EStorageTest>(-5)));
        UNIT_ASSERT(TNegativeEnd::IsValid(static_cast<EStorageTest>(-2)));
        UNIT_ASSERT(!TNegativeEnd::IsValid(static_cast<EStorageTest>(-1)));
    }

    Y_UNIT_TEST(TestHasAnyAndHasAll) {
        const TTestBitSet set(TE_1, TE_3, TE_7);
        UNIT_ASSERT(set.HasAny(TE_1));
        UNIT_ASSERT(!set.HasAny(TE_2));
        UNIT_ASSERT(set.HasAny(TE_2, TE_3));
        UNIT_ASSERT(set.HasAny(TE_2, TE_4, TE_7));
        UNIT_ASSERT(!set.HasAny(TE_2, TE_4, TE_6));
        UNIT_ASSERT(set.HasAll(TE_1));
        UNIT_ASSERT(!set.HasAll(TE_2));
        UNIT_ASSERT(set.HasAll(TE_1, TE_3));
        UNIT_ASSERT(set.HasAll(TE_1, TE_3, TE_7));
        UNIT_ASSERT(!set.HasAll(TE_1, TE_2, TE_7));
        UNIT_ASSERT(set.HasAny(TTestBitSet(TE_2, TE_3)));
        UNIT_ASSERT(set.HasAll(TTestBitSet(TE_1, TE_7)));
        const TTestBitSet empty;
        UNIT_ASSERT(!empty.HasAny(TE_1));
        UNIT_ASSERT(!empty.HasAll(TE_1));
        UNIT_ASSERT(!set.HasAny());
        UNIT_ASSERT(set.HasAll());
        const TSfEnumBitSet<ETestEnum, TE_FIRST, TE_MAX> subset(TE_1);
        UNIT_ASSERT(set.HasAny(subset));
        UNIT_ASSERT(set.HasAll(subset));
    }

    Y_UNIT_TEST(TestMixedHasAnyAndHasAll) {
        struct TDerivedBitSet: TTestBitSet {
            using TTestBitSet::TTestBitSet;
        };
        const TTestBitSet set(TE_1, TE_3, TE_7);
        const TTestBitSet partial(TE_2, TE_3);
        const TTestBitSet disjoint(TE_2, TE_4);
        const TDerivedBitSet subset(TE_1, TE_7);
        const TDerivedBitSet derivedPartial(TE_2, TE_3);
        const TTestBitSet empty;

        UNIT_ASSERT(set.HasAny(TE_2, partial));
        UNIT_ASSERT(set.HasAny(TE_1, disjoint));
        UNIT_ASSERT(!set.HasAny(TE_2, disjoint));
        UNIT_ASSERT(set.HasAll(TE_3, subset));
        UNIT_ASSERT(!set.HasAll(TE_2, subset));
        UNIT_ASSERT(!set.HasAll(TE_1, partial));
        UNIT_ASSERT(set.HasAny(derivedPartial));
        UNIT_ASSERT(set.HasAll(subset));
        UNIT_ASSERT(set.HasAny(TE_2, TE_4, derivedPartial));
        UNIT_ASSERT(set.HasAll(TE_1, TE_3, subset));
        UNIT_ASSERT(set.HasAny(partial, TE_2));
        UNIT_ASSERT(set.HasAll(subset, TE_3));
        UNIT_ASSERT(!set.HasAny(TE_2, empty));
        UNIT_ASSERT(set.HasAll(TE_1, empty));

        UNIT_ASSERT(set.HasAny(TE_1, TThrowingEnumConversion{}, subset));
        UNIT_ASSERT(!set.HasAll(TE_2, TThrowingEnumConversion{}, subset));
        UNIT_ASSERT_EXCEPTION(set.HasAny(TE_2, TThrowingEnumConversion{}, subset), yexception);
        UNIT_ASSERT_EXCEPTION(set.HasAll(TE_1, TThrowingEnumConversion{}, subset), yexception);
    }

    Y_UNIT_TEST(TestVariadicInitialization) {
        TTestBitSet set(TE_1, TE_2, TE_3, TE_4, TE_7);
        UNIT_ASSERT_VALUES_EQUAL(set.Count(), 5);
        UNIT_ASSERT(set.HasAll(TE_1, TE_2, TE_3, TE_4, TE_7));
        set.Init(TE_2, TE_3, TE_5, TE_6);
        UNIT_ASSERT_VALUES_EQUAL(set.Count(), 7);
        set.Init();
        UNIT_ASSERT_VALUES_EQUAL(set.Count(), 7);

        using TSafeSet = TSfEnumBitSet<ETestEnum, TE_FIRST, TE_MAX>;
        TSafeSet safe(TE_1, TE_OVERFLOW, TE_2, TE_OVERFLOW, TE_3);
        UNIT_ASSERT_VALUES_EQUAL(safe.Count(), 3);
        safe.Init(TE_OVERFLOW, TE_4, TE_OVERFLOW, TE_7);
        UNIT_ASSERT_VALUES_EQUAL(safe.Count(), 5);
        UNIT_ASSERT(safe.HasAll(TE_1, TE_2, TE_3, TE_4, TE_7));
        safe.Init();
        UNIT_ASSERT_VALUES_EQUAL(safe.Count(), 5);
        UNIT_ASSERT_EXCEPTION(safe.Init(TE_1, TE_2, TThrowingEnumConversion{}), yexception);

        set.Reset();
        UNIT_ASSERT_EXCEPTION(set.Init(TE_1, TE_2, TE_3, TThrowingEnumConversion{}, TE_7), yexception);
        UNIT_ASSERT_VALUES_EQUAL(set.Count(), 3);
        UNIT_ASSERT(set.HasAll(TE_1, TE_2, TE_3));
        set.Reset();
        set.Init(TE_7);
        UNIT_ASSERT_VALUES_EQUAL(set.Count(), 1);
        UNIT_ASSERT(set.Test(TE_7));
        set.Reset();
        UNIT_ASSERT_EXCEPTION(set.Init(TE_1, TThrowingEnumConversion{}), yexception);
        UNIT_ASSERT_VALUES_EQUAL(set.Count(), 1);
        UNIT_ASSERT(set.Test(TE_1));
    }

    Y_UNIT_TEST(TestExceptionPropagation) {
        TTestBitSet set(TE_1, TE_3);
        UNIT_ASSERT_EXCEPTION(set.HasAny(TThrowingEnumConversion{}), yexception);
        UNIT_ASSERT_EXCEPTION(set.HasAll(TThrowingEnumConversion{}), yexception);
        UNIT_ASSERT_EXCEPTION(TTestBitSet(TE_1, TE_3, TThrowingEnumConversion{}), yexception);

        const ETestEnum values[] = {TE_1, TE_3};
        const ETestEnum* begin = values;
        const ETestEnum* end = values + Y_ARRAY_SIZE(values);
        UNIT_ASSERT(TTestBitSet(begin, end) == set);
        UNIT_ASSERT(!set.TryFromString("invalid"));
    }

    Y_UNIT_TEST(TestConstexprContracts) {
        static_assert(TTestBitSet::BeginIndex == TE_FIRST);
        static_assert(TTestBitSet::EndIndex == TE_MAX);
        static_assert(TTestBitSet::BitsetSize == TE_MAX - TE_FIRST);
        static_assert(TTestBitSet::IsValid(TE_1));
        static_assert(!TTestBitSet::IsValid(TE_OVERFLOW));
    }

    Y_UNIT_TEST(TestEmpty) {
        TTestBitSet mask;
        UNIT_ASSERT(mask.Empty());
        if (mask)
            UNIT_ASSERT(false && "should be empty");

        mask.Set(TE_FIRST);
        UNIT_ASSERT(!mask.Empty());
        UNIT_ASSERT(mask.Count() == 1);
        if (!mask)
            UNIT_ASSERT(false && "should not be empty");
    }

    Y_UNIT_TEST(TestIter) {
        TTestBitSet mask = TTestBitSet(TE_1, TE_3, TE_7);

        TTestBitSet mask2;
        for (auto elem : mask) {
            mask2.Set(elem);
        }

        UNIT_ASSERT(mask == mask2);
    }

    Y_UNIT_TEST(TestSerialization) {
        auto ebs = TTestBitSet(TE_MIDDLE, TE_6, TE_7);

        TStringStream ss;
        ebs.Save(&ss);

        auto ebs2 = TTestBitSet();
        ebs2.Load(&ss);
        UNIT_ASSERT_EQUAL(ebs, ebs2);
    }

    Y_UNIT_TEST(TestLoadIgnoresOutOfRangeValues) {
        using TSet = TEnumBitSet<EStorageTest, 0, 11>;
        const ui32 values[] = {0, 11, 15, 16, 64, ~ui32{0}, 10};
        TStringStream stream;
        ::Save(&stream, static_cast<ui32>(Y_ARRAY_SIZE(values)));
        for (const ui32 value : values) {
            ::Save(&stream, value);
        }
        ::Save(&stream, ui32{1});
        ::Save(&stream, ui32{64});

        TSet set(static_cast<EStorageTest>(5));
        set.Load(&stream);
        UNIT_ASSERT_VALUES_EQUAL(set.Count(), 2);
        UNIT_ASSERT_VALUES_EQUAL(set.Low(), 0x401);

        set.Load(&stream);
        UNIT_ASSERT(set.Empty());
    }

    Y_UNIT_TEST(TestStringRepresentation) {
        auto ebs = TTestBitSet(TE_MIDDLE, TE_6, TE_7);

        UNIT_ASSERT_EQUAL(ebs.ToString(), "D00000000000000000");

        auto ebs2 = TTestBitSet();
        ebs2.FromString("D00000000000000000");
        UNIT_ASSERT_EQUAL(ebs, ebs2);
    }
}
