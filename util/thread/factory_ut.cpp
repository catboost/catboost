#include "factory.h"
#include "pool.h"

#include <library/cpp/testing/unittest/registar.h>

class TThrPoolTest: public TTestBase {
    UNIT_TEST_SUITE(TThrPoolTest);
    UNIT_TEST(TestSystemPool)
    UNIT_TEST(TestAdaptivePool)
    UNIT_TEST_SUITE_END();

    struct TRunAble: public IThreadFactory::IThreadAble {
        inline TRunAble()
            : done(false)
        {
        }

        ~TRunAble() override = default;

        void DoExecute() override {
            done = true;
        }

        bool done;
    };

private:
    inline void TestSystemPool() {
        TRunAble r;

        {
            THolder<IThreadFactory::IThread> thr = SystemThreadFactory()->Run(&r);

            thr->Join();
        }

        UNIT_ASSERT_EQUAL(r.done, true);
    }

    inline void TestAdaptivePool() {
        TRunAble r;

        {
            TAdaptiveThreadPool pool;

            pool.Start(0);

            THolder<IThreadFactory::IThread> thr = pool.Run(&r);

            thr->Join();
        }

        UNIT_ASSERT_EQUAL(r.done, true);
    }
};

UNIT_TEST_SUITE_REGISTRATION(TThrPoolTest);
