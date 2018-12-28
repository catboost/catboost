#include "array_ref.h"

#include <library/unittest/registar.h>

Y_UNIT_TEST_SUITE(TestArrayRef) {
    Y_UNIT_TEST(Test1) {
        TConstArrayRef<char> a("123", 3);
        size_t ret = 0;

        for (char ch : a) {
            ret += ch - '0';
        }

        UNIT_ASSERT_VALUES_EQUAL(ret, 6);
        UNIT_ASSERT((bool)a);
    }

    Y_UNIT_TEST(Test2) {
        char x[] = "123";
        TArrayRef<char> a(x, 3);
        size_t ret = 0;

        for (char& ch : a) {
            ret += ch - '0';
        }

        UNIT_ASSERT_VALUES_EQUAL(ret, 6);
        UNIT_ASSERT((bool)a);

        a.at(0);
    }

    Y_UNIT_TEST(Test3_operator_equal) {
        static constexpr size_t size = 5;
        int a[size]{1, 2, 3, 4, 5};
        int b[size]{5, 4, 3, 2, 1};
        int c[size - 1]{5, 4, 3, 2};
        float d[size]{1.f, 2.f, 3.f, 4.f, 5.f};

        TArrayRef<int> aArr(a);
        TConstArrayRef<int> aConstArr(a, size);

        TArrayRef<int> bArr(b);

        TArrayRef<int> cArr(c, size - 1);

        TArrayRef<float> dArr(d, size);
        TConstArrayRef<float> dConstArr(d, size);

        UNIT_ASSERT_EQUAL(aArr, aConstArr);
        UNIT_ASSERT_EQUAL(dArr, dConstArr);

        UNIT_ASSERT_UNEQUAL(aArr, cArr);
        UNIT_ASSERT_UNEQUAL(aArr, bArr);
    }

    Y_UNIT_TEST(TestArrayRefFromContainer) {
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

    class A {};
    class B {};

    void checkAdl1(TArrayRef<A>) {
    }
    void checkAdl1(TArrayRef<B>) {
    }
    void checkAdl2(TArrayRef<const A>) {
    }
    void checkAdl2(TArrayRef<const B>) {
    }

    Y_UNIT_TEST(TestArrayRefCtorAdl) {
        /* No checks here, the code should simply compile. */

        TVector<A> a;
        TVector<B> b;

        checkAdl1(a);
        checkAdl1(b);

        checkAdl2(a);
        checkAdl2(b);
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
        static_assert(r0.size() == 1, "r0.size() == 1");
        static_assert(r0.data()[0] == 1, "r0.data()[0] == 1");
        static constexpr const TArrayRef<const int> r1{a};
        static_assert(r1.size() == 4, "r1.size() == 4");
        static_assert(r1.data()[3] == -4, "r1.data()[3] == -4");
        static constexpr const TArrayRef<const int> r2 = r1;
        ;
        static_assert(r2.size() == 4, "r2.size() == 4");
        static_assert(r2.data()[2] == -3, "r2.data()[2] == -3");
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
