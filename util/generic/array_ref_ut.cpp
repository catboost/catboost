#include "array_ref.h"

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(TestArrayRef) {
    Y_UNIT_TEST(TestDefaultConstructor) {
        TArrayRef<int> defaulted;
        UNIT_ASSERT_VALUES_EQUAL(defaulted.data(), nullptr);
        UNIT_ASSERT_VALUES_EQUAL(defaulted.size(), 0u);
    }

    Y_UNIT_TEST(TestConstructorFromArray) {
        int x[] = {10, 20, 30};
        TArrayRef<int> ref(x);
        UNIT_ASSERT_VALUES_EQUAL(3u, ref.size());
        UNIT_ASSERT_VALUES_EQUAL(30, ref[2]);
        ref[2] = 50;
        UNIT_ASSERT_VALUES_EQUAL(50, x[2]);

        TArrayRef<const int> constRef(x);
        UNIT_ASSERT_VALUES_EQUAL(3u, constRef.size());
        UNIT_ASSERT_VALUES_EQUAL(50, constRef[2]);
        ref[0] = 100;
        UNIT_ASSERT_VALUES_EQUAL(constRef[0], 100);
    }

    Y_UNIT_TEST(TestAccessingElements) {
        int a[]{1, 2, 3};
        TArrayRef<int> ref(a);

        UNIT_ASSERT_VALUES_EQUAL(ref[0], 1);
        UNIT_ASSERT_VALUES_EQUAL(ref.at(0), 1);

        ref[0] = 5;
        UNIT_ASSERT_VALUES_EQUAL(a[0], 5);

        // FIXME: size checks are implemented via Y_ASSERT, hence there is no way to test them
    }

    Y_UNIT_TEST(TestFrontBack) {
        const int x[] = {1, 2, 3};
        const TArrayRef<const int> rx{x};
        UNIT_ASSERT_VALUES_EQUAL(rx.front(), 1);
        UNIT_ASSERT_VALUES_EQUAL(rx.back(), 3);

        int y[] = {1, 2, 3};
        TArrayRef<int> ry{y};
        UNIT_ASSERT_VALUES_EQUAL(ry.front(), 1);
        UNIT_ASSERT_VALUES_EQUAL(ry.back(), 3);

        ry.front() = 100;
        ry.back() = 500;
        UNIT_ASSERT_VALUES_EQUAL(ry.front(), 100);
        UNIT_ASSERT_VALUES_EQUAL(ry.back(), 500);
        UNIT_ASSERT_VALUES_EQUAL(y[0], 100);
        UNIT_ASSERT_VALUES_EQUAL(y[2], 500);
    }

    Y_UNIT_TEST(TestIterator) {
        int array[] = {17, 19, 21};
        TArrayRef<int> r(array, 3);

        TArrayRef<int>::iterator iterator = r.begin();
        for (auto& i : array) {
            UNIT_ASSERT(iterator != r.end());
            UNIT_ASSERT_VALUES_EQUAL(i, *iterator);
            ++iterator;
        }
        UNIT_ASSERT(iterator == r.end());
    }

    Y_UNIT_TEST(TestReverseIterators) {
        const int x[] = {1, 2, 3};
        const TArrayRef<const int> rx{x};
        auto i = rx.crbegin();
        UNIT_ASSERT_VALUES_EQUAL(*i, 3);
        ++i;
        UNIT_ASSERT_VALUES_EQUAL(*i, 2);
        ++i;
        UNIT_ASSERT_VALUES_EQUAL(*i, 1);
        ++i;
        UNIT_ASSERT_EQUAL(i, rx.crend());
    }

    Y_UNIT_TEST(TestConstIterators) {
        int x[] = {1, 2, 3};
        TArrayRef<int> rx{x};
        UNIT_ASSERT_EQUAL(rx.begin(), rx.cbegin());
        UNIT_ASSERT_EQUAL(rx.end(), rx.cend());
        UNIT_ASSERT_EQUAL(rx.rbegin(), rx.crbegin());
        UNIT_ASSERT_EQUAL(rx.rend(), rx.crend());

        int w[] = {1, 2, 3};
        const TArrayRef<int> rw{w};
        UNIT_ASSERT_EQUAL(rw.begin(), rw.cbegin());
        UNIT_ASSERT_EQUAL(rw.end(), rw.cend());
        UNIT_ASSERT_EQUAL(rw.rbegin(), rw.crbegin());
        UNIT_ASSERT_EQUAL(rw.rend(), rw.crend());

        int y[] = {1, 2, 3};
        TArrayRef<const int> ry{y};
        UNIT_ASSERT_EQUAL(ry.begin(), ry.cbegin());
        UNIT_ASSERT_EQUAL(ry.end(), ry.cend());
        UNIT_ASSERT_EQUAL(ry.rbegin(), ry.crbegin());
        UNIT_ASSERT_EQUAL(ry.rend(), ry.crend());

        const int z[] = {1, 2, 3};
        TArrayRef<const int> rz{z};
        UNIT_ASSERT_EQUAL(rz.begin(), rz.cbegin());
        UNIT_ASSERT_EQUAL(rz.end(), rz.cend());
        UNIT_ASSERT_EQUAL(rz.rbegin(), rz.crbegin());
        UNIT_ASSERT_EQUAL(rz.rend(), rz.crend());

        const int q[] = {1, 2, 3};
        const TArrayRef<const int> rq{q};
        UNIT_ASSERT_EQUAL(rq.begin(), rq.cbegin());
        UNIT_ASSERT_EQUAL(rq.end(), rq.cend());
        UNIT_ASSERT_EQUAL(rq.rbegin(), rq.crbegin());
        UNIT_ASSERT_EQUAL(rq.rend(), rq.crend());
    }

    Y_UNIT_TEST(TestCreatingFromStringLiteral) {
        TConstArrayRef<char> knownSizeRef("123", 3);
        size_t ret = 0;

        for (char ch : knownSizeRef) {
            ret += ch - '0';
        }

        UNIT_ASSERT_VALUES_EQUAL(ret, 6);
        UNIT_ASSERT_VALUES_EQUAL(knownSizeRef.size(), 3);
        UNIT_ASSERT_VALUES_EQUAL(knownSizeRef.at(0), '1');

        /*
         * When TArrayRef is being constructed from string literal,
         * trailing zero will be added into it.
         */
        TConstArrayRef<char> autoSizeRef("456");
        UNIT_ASSERT_VALUES_EQUAL(autoSizeRef[0], '4');
        UNIT_ASSERT_VALUES_EQUAL(autoSizeRef[3], '\0');
    }

    Y_UNIT_TEST(TestEqualityOperator) {
        static constexpr size_t size = 5;
        int a[size]{1, 2, 3, 4, 5};
        int b[size]{5, 4, 3, 2, 1};
        int c[size - 1]{5, 4, 3, 2};
        float d[size]{1.f, 2.f, 3.f, 4.f, 5.f};

        TArrayRef<int> aRef(a);
        TConstArrayRef<int> aConstRef(a, size);

        TArrayRef<int> bRef(b);

        TArrayRef<int> cRef(c, size - 1);

        TArrayRef<float> dRef(d, size);
        TConstArrayRef<float> dConstRef(d, size);

        UNIT_ASSERT_EQUAL(aRef, aConstRef);
        UNIT_ASSERT_EQUAL(dRef, dConstRef);

        UNIT_ASSERT_UNEQUAL(aRef, cRef);
        UNIT_ASSERT_UNEQUAL(aRef, bRef);

        TArrayRef<int> bSubRef(b, size - 1);

        // Testing if operator== compares values, not pointers
        UNIT_ASSERT_EQUAL(cRef, bSubRef);
    }

    Y_UNIT_TEST(TestImplicitConstructionFromContainer) {
        /* Just test compilation. */
        auto fc = [](TArrayRef<const int>) {};
        auto fm = [](TArrayRef<int>) {};

        fc(TVector<int>({1}));

        const TVector<int> ac = {1};
        TVector<int> am = {1};

        fc(ac);
        fc(am);
        fm(am);
        // fm(ac); // This one shouldn't compile.
    }

    Y_UNIT_TEST(TestFirstLastSubspan) {
        const int arr[] = {1, 2, 3, 4, 5};
        TArrayRef<const int> aRef(arr);

        UNIT_ASSERT_EQUAL(aRef.first(2), MakeArrayRef(std::vector<int>{1, 2}));
        UNIT_ASSERT_EQUAL(aRef.last(2), MakeArrayRef(std::vector<int>{4, 5}));
        UNIT_ASSERT_EQUAL(aRef.subspan(2), MakeArrayRef(std::vector<int>{3, 4, 5}));
        UNIT_ASSERT_EQUAL(aRef.subspan(1, 3), MakeArrayRef(std::vector<int>{2, 3, 4}));
    }

    Y_UNIT_TEST(TestSlice) {
        const int a0[] = {1, 2, 3};
        TArrayRef<const int> r0(a0);
        TArrayRef<const int> s0 = r0.Slice(2);

        UNIT_ASSERT_VALUES_EQUAL(s0.size(), 1);
        UNIT_ASSERT_VALUES_EQUAL(s0[0], 3);

        const int a1[] = {1, 2, 3, 4};
        TArrayRef<const int> r1(a1);
        TArrayRef<const int> s1 = r1.Slice(2, 1);

        UNIT_ASSERT_VALUES_EQUAL(s1.size(), 1);
        UNIT_ASSERT_VALUES_EQUAL(s1[0], 3);

        // FIXME: size checks are implemented via Y_ASSERT, hence there is no way to test them
    }

    Y_UNIT_TEST(SubRegion) {
        TVector<char> x;
        for (size_t i = 0; i < 42; ++i) {
            x.push_back('a' + (i * 42424243) % 13);
        }
        TArrayRef<const char> ref(x.data(), 42);
        for (size_t i = 0; i <= 50; ++i) {
            TVector<char> expected;
            for (size_t j = 0; j <= 100; ++j) {
                UNIT_ASSERT(MakeArrayRef(expected) == ref.SubRegion(i, j));
                if (i + j < 42) {
                    expected.push_back(x[i + j]);
                }
            }
        }
    }

    Y_UNIT_TEST(TestAsBytes) {
        const int16_t constArr[] = {1, 2, 3};
        TArrayRef<const int16_t> constRef(constArr);
        auto bytesRef = as_bytes(constRef);

        UNIT_ASSERT_VALUES_EQUAL(bytesRef.size(), sizeof(int16_t) * constRef.size());
        UNIT_ASSERT_EQUAL(
            bytesRef,
            MakeArrayRef(std::vector<char>{0x01, 0x00, 0x02, 0x00, 0x03, 0x00}));

        // should not compile
        // as_writable_bytes(constRef);
    }

    Y_UNIT_TEST(TestAsWritableBytes) {
        uint32_t uintArr[] = {0x0c'00'0d'0e};
        TArrayRef<uint32_t> uintRef(uintArr);
        auto writableBytesRef = as_writable_bytes(uintRef);

        UNIT_ASSERT_VALUES_EQUAL(writableBytesRef.size(), sizeof(uint32_t));
        UNIT_ASSERT_EQUAL(
            writableBytesRef,
            MakeArrayRef(std::vector<char>{0x0e, 0x0d, 0x00, 0x0c}));

        uint32_t newVal = 0xde'ad'be'ef;
        std::memcpy(writableBytesRef.data(), &newVal, writableBytesRef.size());
        UNIT_ASSERT_VALUES_EQUAL(uintArr[0], newVal);
    }

    Y_UNIT_TEST(TestTypeDeductionViaMakeArrayRef) {
        TVector<int> vec{17, 19, 21};
        TArrayRef<int> ref = MakeArrayRef(vec);
        UNIT_ASSERT_VALUES_EQUAL(21, ref[2]);
        ref[1] = 23;
        UNIT_ASSERT_VALUES_EQUAL(23, vec[1]);

        const TVector<int>& constVec(vec);
        TArrayRef<const int> constRef = MakeArrayRef(constVec);
        UNIT_ASSERT_VALUES_EQUAL(21, constRef[2]);

        TArrayRef<const int> constRefFromNonConst = MakeArrayRef(vec);
        UNIT_ASSERT_VALUES_EQUAL(23, constRefFromNonConst[1]);
    }

    static void Do(const TArrayRef<int> a) {
        a[0] = 8;
    }

    Y_UNIT_TEST(TestConst) {
        int a[] = {1, 2};
        Do(a);
        UNIT_ASSERT_VALUES_EQUAL(a[0], 8);
    }

    Y_UNIT_TEST(TestConstexpr) {
        static constexpr const int a[] = {1, 2, -3, -4};
        static constexpr const auto r0 = MakeArrayRef(a, 1);
        static_assert(r0.size() == 1, "r0.size() is not equal 1");
        static_assert(r0.data()[0] == 1, "r0.data()[0] is not equal to 1");

        static constexpr const TArrayRef<const int> r1{a};
        static_assert(r1.size() == 4, "r1.size() is not equal to 4");
        static_assert(r1.data()[3] == -4, "r1.data()[3] is not equal to -4");

        static constexpr const TArrayRef<const int> r2 = r1;
        static_assert(r2.size() == 4, "r2.size() is not equal to 4");
        static_assert(r2.data()[2] == -3, "r2.data()[2] is not equal to -3");
    }

    template <typename T>
    static void Foo(const TConstArrayRef<T>) {
        // noop
    }

    Y_UNIT_TEST(TestMakeConstArrayRef) {
        TVector<int> data;

        // Won't compile because can't deduce `T` for `Foo`
        // Foo(data);

        // Won't compile because again can't deduce `T` for `Foo`
        // Foo(MakeArrayRef(data));

        // Success!
        Foo(MakeConstArrayRef(data));

        const TVector<int> constData;
        Foo(MakeConstArrayRef(constData));
    }
}
