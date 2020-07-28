#include "smallobj.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/hash_set.h>

class TSmallObjAllocTest: public TTestBase {
    struct TClass: public TObjectFromPool<TClass> {
        char buf[125];
    };

    struct TErrClass: public TObjectFromPool<TErrClass> {
        inline TErrClass(bool t) {
            if (t) {
                throw 1;
            }
        }
    };

    struct TClass64: public TObjectFromPool<TClass64> {
        alignas(64) ui64 Data = 0;
    };

    UNIT_TEST_SUITE(TSmallObjAllocTest);
    UNIT_TEST(TestAlign)
    UNIT_TEST(TestError)
    UNIT_TEST(TestAllocate)
    UNIT_TEST_SUITE_END();

private:
    void TestAlign() {
        TClass64::TPool pool(TDefaultAllocator::Instance());
        TClass64* f1 = new (&pool) TClass64;
        TClass64* f2 = new (&pool) TClass64;
        TClass64* f3 = new (&pool) TClass64;
        TClass64* f4 = new (&pool) TClass64;
        UNIT_ASSERT_VALUES_EQUAL(64u, alignof(TClass64));
        UNIT_ASSERT_VALUES_EQUAL((size_t)0, (size_t)(f1) & (alignof(TClass64) - 1));
        UNIT_ASSERT_VALUES_EQUAL((size_t)0, (size_t)(f2) & (alignof(TClass64) - 1));
        UNIT_ASSERT_VALUES_EQUAL((size_t)0, (size_t)(f3) & (alignof(TClass64) - 1));
        UNIT_ASSERT_VALUES_EQUAL((size_t)0, (size_t)(f4) & (alignof(TClass64) - 1));
    }

    inline void TestError() {
        TErrClass::TPool pool(TDefaultAllocator::Instance());
        TErrClass* f = new (&pool) TErrClass(false);

        delete f;

        bool wasError = false;

        try {
            new (&pool) TErrClass(true);
        } catch (...) {
            wasError = true;
        }

        UNIT_ASSERT(wasError);
        UNIT_ASSERT_EQUAL(f, new (&pool) TErrClass(false));
    }

    inline void TestAllocate() {
        TClass::TPool pool(TDefaultAllocator::Instance());
        THashSet<TClass*> alloced;

        for (size_t i = 0; i < 10000; ++i) {
            TClass* c = new (&pool) TClass;

            UNIT_ASSERT_EQUAL(alloced.find(c), alloced.end());
            alloced.insert(c);
        }

        for (auto it : alloced) {
            delete it;
        }

        for (size_t i = 0; i < 10000; ++i) {
            TClass* c = new (&pool) TClass;

            UNIT_ASSERT(alloced.find(c) != alloced.end());
        }

        UNIT_ASSERT_EQUAL(alloced.find(new (&pool) TClass), alloced.end());
    }
};

UNIT_TEST_SUITE_REGISTRATION(TSmallObjAllocTest);
