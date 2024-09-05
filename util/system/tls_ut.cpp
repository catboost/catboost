#include "tls.h"
#include "thread.h"

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(TTestTLS) {
    struct X {
        inline X()
            : V(0)
        {
        }

        inline void Do() {
            ++TlsRef(V);
        }

        inline int Get() {
            return TlsRef(V);
        }

        Y_THREAD(int)
        V;
    };

    Y_UNIT_TEST(TestHugeSetup) {
        TArrayHolder<X> x(new X[100000]);

        struct TThr: public ISimpleThread {
            inline TThr(X* ptr)
                : P(ptr)
            {
            }

            void* ThreadProc() noexcept override {
                for (size_t i = 0; i < 100000; ++i) {
                    P[i].Do();
                }

                return nullptr;
            }

            X* P;
        };

        TThr thr1(x.Get());
        TThr thr2(x.Get());

        thr1.Start();
        thr2.Start();

        thr1.Join();
        thr2.Join();

        for (size_t i = 0; i < 100000; ++i) {
            UNIT_ASSERT_VALUES_EQUAL(x.Get()[i].Get(), 0);
        }
    }
} // Y_UNIT_TEST_SUITE(TTestTLS)
