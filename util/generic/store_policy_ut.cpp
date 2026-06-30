#include "store_policy.h"

#include <library/cpp/testing/unittest/registar.h>
#include <util/generic/vector.h>

Y_UNIT_TEST_SUITE(StorePolicy) {
    Y_UNIT_TEST(Compileability) {
        // construction
        TAutoEmbedOrPtrPolicy<THolder<int>>(MakeHolder<int>(1));
        TAutoEmbedOrPtrPolicy<TVector<int>>(TVector<int>{1, 2, 3});
        auto a = MakeHolder<int>(42);
        TAutoEmbedOrPtrPolicy<THolder<int>&>{a};

        // const
        (**TAutoEmbedOrPtrPolicy<THolder<int>>(MakeHolder<int>(1)).Ptr())++; // ok
        (**TAutoEmbedOrPtrPolicy<THolder<int>&>(a).Ptr())++;                 // ok

        const TVector<int> b = {0};
        auto bValue = (*TAutoEmbedOrPtrPolicy<const TVector<int>&>(b).Ptr())[0]; // ok
        // (*TAutoEmbedOrPtrPolicy<const TVector<int>&>(b).Ptr())[0]++; // not ok
        Y_UNUSED(bValue);
    }

    template <typename T, typename TFunc>
    void FunctionTakingRefDefaultIsObject(T&& a, TFunc func) {
        TAutoEmbedOrPtrPolicy<T> refHolder(a);
        func(refHolder);
    }

    Y_UNIT_TEST(Reference) {
        {
            TVector<ui32> a = {1, 2, 3};

            FunctionTakingRefDefaultIsObject(a, [](auto& holder) {
                holder.Ptr()->push_back(4);
                auto secondHolder = holder;
                secondHolder.Ptr()->push_back(5);
            });

            UNIT_ASSERT_VALUES_EQUAL(a.size(), 5);
        }
        {
            const TVector<ui32> a = {1, 2, 3};

            static_assert(std::is_const<decltype(a)>::value);

            FunctionTakingRefDefaultIsObject(a, [](auto& holder) {
                static_assert(std::is_const<std::remove_reference_t<decltype(*holder.Ptr())>>::value);
                UNIT_ASSERT_VALUES_EQUAL(holder.Ptr()->size(), 3);
            });
        }
    }

    template <typename T, typename TFunc>
    void FunctionTakingObjectDefaultObject(T&& a, TFunc func) {
        TAutoEmbedOrPtrPolicy<T> objectHolder(std::forward<T>(a));
        func(objectHolder);
    }

    Y_UNIT_TEST(Object) {
        TVector<ui32> a = {1, 2, 3};

        FunctionTakingObjectDefaultObject(std::move(a), [&a](auto& holder) {
            static_assert(std::is_copy_assignable<decltype(holder)>::value);
            UNIT_ASSERT_VALUES_EQUAL(a.size(), 0);
            UNIT_ASSERT_VALUES_EQUAL(holder.Ptr()->size(), 3);
            holder.Ptr()->push_back(4);
            auto secondHolder = holder;
            secondHolder.Ptr()->push_back(5);

            UNIT_ASSERT_VALUES_EQUAL(holder.Ptr()->size(), 4);
            UNIT_ASSERT_VALUES_EQUAL(secondHolder.Ptr()->size(), 5);
        });

        UNIT_ASSERT_VALUES_EQUAL(a.size(), 0);

        THolder<int> b = MakeHolder<int>(42);
        FunctionTakingObjectDefaultObject(std::move(b), [](auto& holder) {
            static_assert(!std::is_copy_assignable<decltype(holder)>::value);
            UNIT_ASSERT_VALUES_EQUAL(**holder.Ptr(), 42);
            auto secondHolder = std::move(holder);
            UNIT_ASSERT(!*holder.Ptr());
            UNIT_ASSERT_VALUES_EQUAL(**secondHolder.Ptr(), 42);
        });
    }

    struct TNoDefaultConstructible {
        explicit TNoDefaultConstructible(int) noexcept {
        }
    };

    template <class TType, class TBaseType>
    static void TestStoryPolicyConstructors() {
        if constexpr (std::is_default_constructible_v<TType>) {
            TType instance{};
            Y_UNUSED(instance);
        }
        UNIT_ASSERT_VALUES_EQUAL(std::is_default_constructible_v<TType>, std::is_default_constructible_v<TBaseType>);
        if constexpr (std::is_constructible_v<TType, int>) {
            TType instance{4};
            Y_UNUSED(instance);
        }
        UNIT_ASSERT_VALUES_EQUAL((std::is_constructible_v<TType, int>), (std::is_constructible_v<TBaseType, int>));
    }

    template <class TBaseType>
    static void TestWrapperConstructors() {
        TestStoryPolicyConstructors<TWithRefCount<TBaseType, TAtomicCounter>, TBaseType>();
        TestStoryPolicyConstructors<TEmbedPolicy<TBaseType>, TBaseType>();
        TestStoryPolicyConstructors<TSimpleRefPolicy<TBaseType>, TBaseType>();
    }

    Y_UNIT_TEST(ConstructorTraits) {
        TestWrapperConstructors<TNoDefaultConstructible>();
        TestWrapperConstructors<TVector<short>>();
    }
} // Y_UNIT_TEST_SUITE(StorePolicy)
