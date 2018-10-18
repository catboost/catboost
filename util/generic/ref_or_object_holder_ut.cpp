#include "ref_or_object_holder.h"

#include <library/unittest/registar.h>
#include <util/generic/algorithm.h>
#include <util/generic/vector.h>

Y_UNIT_TEST_SUITE(RefOrObjectHolder) {

    template <typename T, typename TFunc>
    void FunctionTakingRefDefaultIsObject(T&& a, TFunc func) {
        static_assert(std::is_pointer<typename TRefOrObjectHolder<T>::TObjectStorage>::value);
        TRefOrObjectHolder<T> refHolder(a);
        func(refHolder);
    }

    template <typename T, typename TFunc>
    void FunctionTakingRefDefaultIsSharedPtr(T&& a, TFunc func) {
        static_assert(std::is_pointer<typename TRefOrObjectHolder<T>::TObjectStorage>::value);
        TRefOrObjectHolder<T> refHolder(a);
        func(refHolder);
    }

    Y_UNIT_TEST(Reference) {
        {
            TVector<ui32> a = {1, 2, 3};

            FunctionTakingRefDefaultIsObject(a, [](auto& holder) {
                holder->push_back(4);
                auto secondHolder = holder;
                secondHolder->push_back(5);
            });

            UNIT_ASSERT_VALUES_EQUAL(a.size(), 5);
        }
        {
            const TVector<ui32> a = {1, 2, 3};

            static_assert(std::is_const<decltype(a)>::value);

            FunctionTakingRefDefaultIsObject(a, [](auto& holder) {
                using TConcreteRefOfObjectHolder = typename std::remove_reference<decltype(holder)>::type;
                static_assert(std::is_const<typename TConcreteRefOfObjectHolder::TObject>::value);
                UNIT_ASSERT_VALUES_EQUAL(holder->size(), 3);
            });

        }
        {
            TVector<ui32> a = {1, 2, 3};

            FunctionTakingRefDefaultIsSharedPtr(a, [](auto& holder) {
                holder->push_back(4);
                auto secondHolder = holder;
                secondHolder->push_back(5);
            });

            UNIT_ASSERT_VALUES_EQUAL(a.size(), 5);
        }
    }

    template <typename T, typename TFunc>
    void FunctionTakingObjectDefaultObject(T&& a, TFunc func) {
        TRefOrObjectHolder<T> objectHolder(a);
        func(objectHolder);
    }

    Y_UNIT_TEST(Object) {
        TVector<ui32> a = {1, 2, 3};

        FunctionTakingObjectDefaultObject(std::move(a), [&a](auto& holder) {
            static_assert(std::is_copy_assignable<decltype(holder)>::value);
            UNIT_ASSERT_VALUES_EQUAL(a.size(), 0);
            UNIT_ASSERT_VALUES_EQUAL(holder->size(), 3);
            holder->push_back(4);
            auto secondHolder = holder;
            secondHolder->push_back(5);

            UNIT_ASSERT_VALUES_EQUAL(holder->size(), 4);
            UNIT_ASSERT_VALUES_EQUAL(secondHolder->size(), 5);
        });

        UNIT_ASSERT_VALUES_EQUAL(a.size(), 0);

        THolder<int> b = MakeHolder<int>(42);
        FunctionTakingObjectDefaultObject(std::move(b), [](auto& holder) {
            static_assert(!std::is_copy_assignable<decltype(holder)>::value);
            UNIT_ASSERT_VALUES_EQUAL(**holder, 42);
            auto secondHolder = std::move(holder);
            UNIT_ASSERT(!*holder);
            UNIT_ASSERT_VALUES_EQUAL(**secondHolder, 42);
        });

    }

    template <typename T, typename TFunc>
    void FunctionTakingObjectDefaultSharedPtr(T&& a, TFunc func) {
        TRefOrObjectSharedHolder<T> objectHolder(a);
        func(objectHolder);
    }

    Y_UNIT_TEST(SharedPtr) {
        TVector<ui32> a = {1, 2, 3};

        FunctionTakingObjectDefaultSharedPtr(std::move(a), [&a](auto& holder) {
            static_assert(std::is_copy_assignable<decltype(a)>::value);
            UNIT_ASSERT_VALUES_EQUAL(a.size(), 0);
            UNIT_ASSERT_VALUES_EQUAL(holder->size(), 3);
            holder->push_back(4);
            auto secondHolder = holder;
            secondHolder->push_back(5);

            UNIT_ASSERT_VALUES_EQUAL(holder->size(), 5);
            UNIT_ASSERT_VALUES_EQUAL(secondHolder->size(), 5);
        });

        UNIT_ASSERT_VALUES_EQUAL(a.size(), 0);

        THolder<int> b = MakeHolder<int>(42);
        FunctionTakingObjectDefaultSharedPtr(std::move(b), [](auto& holder) {
            static_assert(std::is_copy_assignable<decltype(holder)>::value);
            UNIT_ASSERT_VALUES_EQUAL(**holder, 42);
            auto secondHolder = holder;
            UNIT_ASSERT_VALUES_EQUAL(**holder, 42);
            UNIT_ASSERT_VALUES_EQUAL(**secondHolder, 42);
        });

    }

}
