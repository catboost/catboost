#include "object_counter.h"

#include <library/unittest/registar.h>

SIMPLE_UNIT_TEST_SUITE(ObjectsCounter) {
    struct TObject: public TObjectCounter<TObject> {
    };

    SIMPLE_UNIT_TEST(Test1) {
        TObject obj;
        yvector<TObject> objects;
        for (ui32 i = 0; i < 100; ++i) {
            objects.push_back(obj);
        }
        UNIT_ASSERT_EQUAL(TObjectCounter<TObject>::ObjectCount(), 101);
    }

    SIMPLE_UNIT_TEST(TestEq) {
        TObject obj;
        {
            TObject obj1 = obj;
            UNIT_ASSERT_EQUAL(TObjectCounter<TObject>::ObjectCount(), 2);
        }
        UNIT_ASSERT_EQUAL(TObjectCounter<TObject>::ObjectCount(), 1);
    }

    SIMPLE_UNIT_TEST(TestMove) {
        TObject obj;
        UNIT_ASSERT_EQUAL(TObjectCounter<TObject>::ObjectCount(), 1);
        {
            TObject obj1 = std::move(obj);
            UNIT_ASSERT_EQUAL(TObjectCounter<TObject>::ObjectCount(), 2);
        }
        UNIT_ASSERT_EQUAL(TObjectCounter<TObject>::ObjectCount(), 1);
    }
}
