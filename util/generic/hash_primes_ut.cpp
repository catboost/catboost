#include "hash_primes.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/vector.h>
#include <util/string/builder.h>
#include <util/random/fast.h>

Y_UNIT_TEST_SUITE(TestHashPrimes) {
    Y_UNIT_TEST(Test1) {
        UNIT_ASSERT_VALUES_EQUAL(HashBucketCount(1), 7);
        UNIT_ASSERT_VALUES_EQUAL(HashBucketCount(6), 7);
        UNIT_ASSERT_VALUES_EQUAL(HashBucketCount(7), 7);
        UNIT_ASSERT_VALUES_EQUAL(HashBucketCount(8), 17);
    }

    static TVector<size_t> Numbers() {
        TVector<size_t> numbers;

        TFastRng64 rng{961923};
        size_t k = 1;
        for (size_t j = 0; j < 8000; ++j) {
            numbers.push_back(rng.GenRand());
            numbers.push_back(k *= 57);
        }
        for (size_t p = 1; p != 0; p <<= 1) {
            for (size_t offset : {-2, -1, 0, 1, 2}) {
                numbers.push_back(p + offset);
            }
        }
        return numbers;
    }

    static TVector<size_t> Divisors() {
        TVector<size_t> divisors;
        divisors.push_back(HashBucketCountExt(0)());
        for (;;) {
            const size_t prevSize = divisors.back();
            const size_t nextSize = HashBucketCountExt(prevSize + 1)();
            if (nextSize <= prevSize) {
                break;
            }
            divisors.push_back(nextSize);
        }
        return divisors;
    }

    Y_UNIT_TEST(Remainder) {
        const TVector<size_t> numbers = Numbers();
        const TVector<size_t> divisors = Divisors();

        auto testDivisor = [&](const auto& c) {
            for (size_t n : numbers) {
                UNIT_ASSERT_VALUES_EQUAL_C(n % c(), c.Remainder(n), (TStringBuilder() << "n=" << n << "; d=" << c()));
            }
        };

        for (size_t d : divisors) {
            const auto c = HashBucketCountExt(d);
            UNIT_ASSERT_VALUES_EQUAL_C(d, c(), (TStringBuilder() << "d=" << d));
            testDivisor(c);
        }
        testDivisor(::NPrivate::THashDivisor::One());
    }

    Y_UNIT_TEST(MisleadingHints) {
        TFastRng64 rng{332142};
        TVector<size_t> cases = Numbers();
        for (size_t d : Divisors()) {
            cases.push_back(d);
        }

        for (size_t c : cases) {
            for (size_t reps = 0; reps < 3; ++reps) {
                const i8 hint = rng.Uniform(256) - 128;
                UNIT_ASSERT_VALUES_EQUAL_C(HashBucketCountExt(c)(), HashBucketCountExt(c, hint)(), (TStringBuilder() << "c=" << c << "; hint=" << hint));
            }
        }
    }
} // Y_UNIT_TEST_SUITE(TestHashPrimes)
