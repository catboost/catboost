#include "function_ref.h"

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(TestFunctionRef) {
    template <typename Signature>
    struct TTestFunction;

    template <typename Ret, typename... Args, bool IsNoexcept>
    struct TTestFunction<Ret(Args...) noexcept(IsNoexcept)> {
        Ret operator()(Args...) const noexcept(IsNoexcept) {
            return {};
        }
    };

    Y_UNIT_TEST(NonDefaultConstructible) {
        static_assert(!std::is_default_constructible_v<TFunctionRef<void()>>);
        static_assert(!std::is_default_constructible_v<TFunctionRef<void() noexcept>>);
        static_assert(!std::is_default_constructible_v<TFunctionRef<int(double, void********* megaptr, TTestFunction<void(int)>)>>);
    }

    int F1(bool x) {
        if (x) {
            throw 19;
        }
        return 42;
    }

    int F2(bool x) noexcept {
        return 42 + x;
    }

    static const TTestFunction<int(bool)> C1;
    static const TTestFunction<int(bool) noexcept> C2;

    Y_UNIT_TEST(Noexcept) {
        static_assert(std::is_constructible_v<TFunctionRef<int(bool)>, decltype(F1)>);
        static_assert(std::is_constructible_v<TFunctionRef<int(bool)>, decltype(F2)>);
        static_assert(!std::is_constructible_v<TFunctionRef<int(bool) noexcept>, decltype(F1)>);
        static_assert(std::is_constructible_v<TFunctionRef<int(bool) noexcept>, decltype(F2)>);

        static_assert(std::is_constructible_v<TFunctionRef<int(bool)>, decltype(C1)>);
        static_assert(std::is_constructible_v<TFunctionRef<int(bool)>, decltype(C2)>);
        static_assert(!std::is_constructible_v<TFunctionRef<int(bool) noexcept>, decltype(C1)>);
        static_assert(std::is_constructible_v<TFunctionRef<int(bool) noexcept>, decltype(C2)>);
    }

    Y_UNIT_TEST(Deduction) {
        TFunctionRef ref1(F1);
        TFunctionRef ref2(F2);
        TFunctionRef ref3(C1);
        TFunctionRef ref4(C2);

        static_assert(!std::is_nothrow_invocable_r_v<int, decltype(ref1), bool>);
        static_assert(std::is_nothrow_invocable_r_v<int, decltype(ref2), bool>);
        static_assert(std::is_same_v<decltype(ref1)::TSignature, int(bool)>);
        static_assert(std::is_same_v<decltype(ref2)::TSignature, int(bool) noexcept>);
    }

    void WithCallback(TFunctionRef<double(double, int) noexcept>);

    void Iterate(int from, int to, TFunctionRef<void(int)> callback) {
        while (from < to) {
            callback(from++);
        }
    }

    void IterateNoexcept(int from, int to, TFunctionRef<void(int) noexcept> callback) {
        while (from < to) {
            callback(from++);
        }
    }

    Y_UNIT_TEST(AsArgument) {
        int sum = 0;
        Iterate(0, 10, [&](int x) { sum += x; });
        UNIT_ASSERT_EQUAL(sum, 45);

        Iterate(0, 10, [&](int x) noexcept { sum += x; });
        UNIT_ASSERT_EQUAL(sum, 90);

        IterateNoexcept(0, 10, [&](int x) noexcept { sum += x; });
        UNIT_ASSERT_EQUAL(sum, 135);

        auto summer = [&](int x) { sum += x; };
        Iterate(0, 10, summer);
        Iterate(0, 10, summer);
        Iterate(0, 10, summer);
        UNIT_ASSERT_EQUAL(sum, 270);

        TFunctionRef ref = summer;
        Iterate(0, 10, ref);
        UNIT_ASSERT_EQUAL(sum, 315);
    }

    int GlobalSum = 0;
    void AddToGlobalSum(int x) {
        GlobalSum += x;
    }

    Y_UNIT_TEST(FunctionPointer) {
        GlobalSum = 0;
        Iterate(0, 10, AddToGlobalSum);
        UNIT_ASSERT_EQUAL(GlobalSum, 45);

        TFunctionRef ref1 = AddToGlobalSum;
        Iterate(0, 10, ref1);
        UNIT_ASSERT_EQUAL(GlobalSum, 90);

        TFunctionRef ref2{AddToGlobalSum};
        Iterate(0, 10, ref2);
        UNIT_ASSERT_EQUAL(GlobalSum, 135);
    }

    Y_UNIT_TEST(Reassign) {
        TFunctionRef kek = [](double) { return 42; };
        kek = [](double) { return 19; };
        kek = [](int) { return 22.8; };
    }

    const char* Greet() {
        return "Hello, world!";
    }

    Y_UNIT_TEST(ImplicitCasts) {
        TFunctionRef<void(int)> ref = [](int x) { return x; };
        ref = [](double x) { return x; };
        ref = [](char x) { return x; };

        TFunctionRef<int()> ref1 = [] { return 0.5; };
        ref1 = [] { return 'a'; };
        ref1 = [] { return 124u; };

        TFunctionRef<TStringBuf()> ref2{Greet};
    }

    Y_UNIT_TEST(StatelessLambdaLifetime) {
        TFunctionRef<int(int, int)> ref{[](int a, int b) { return a + b; }};
        UNIT_ASSERT_EQUAL(ref(5, 5), 10);
    }

    Y_UNIT_TEST(ForwardArguments) {
        char x = 'x';
        TFunctionRef<void(std::unique_ptr<int>, char&)> ref = [](std::unique_ptr<int> ptr, char& ch) {
            UNIT_ASSERT_EQUAL(*ptr, 5);
            ch = 'a';
        };
        ref(std::make_unique<int>(5), x);
        UNIT_ASSERT_EQUAL(x, 'a');
    }
} // Y_UNIT_TEST_SUITE(TestFunctionRef)
