#include "adaptor.h"
#include "yexception.h"

#include <library/unittest/registar.h>

struct TOnCopy : yexception {
};

struct TOnMove : yexception {
};

struct TState {
    explicit TState() {
    }

    TState(const TState&) {
        ythrow TOnCopy();
    }

    TState(TState&&) {
        ythrow TOnMove();
    }

    void operator=(const TState&) {
        ythrow TOnCopy();
    }

    void rbegin() const {
    }

    void rend() const {
    }
};

SIMPLE_UNIT_TEST_SUITE(TReverseAdaptor) {
    SIMPLE_UNIT_TEST(ReadTest) {
        yvector<int> cont = {1, 2, 3};
        yvector<int> etalon = {3, 2, 1};
        size_t idx = 0;
        for (const auto& x : Reversed(cont)) {
            UNIT_ASSERT_VALUES_EQUAL(etalon[idx++], x);
        }
    }

    SIMPLE_UNIT_TEST(WriteTest) {
        yvector<int> cont = {1, 2, 3};
        yvector<int> etalon = {3, 6, 9};
        size_t idx = 0;
        for (auto& x : Reversed(cont)) {
            x *= x + idx++;
        }
        idx = 0;
        for (auto& x : cont) {
            UNIT_ASSERT_VALUES_EQUAL(etalon[idx++], x);
        }
    }

    SIMPLE_UNIT_TEST(InnerTypeTest) {
        using TStub = yvector<int>;
        TStub stub;
        const TStub cstub;

        using namespace NPrivate;
        UNIT_ASSERT_TYPES_EQUAL(decltype(Reversed(stub)), TReverseImpl<TStub&>);
        UNIT_ASSERT_TYPES_EQUAL(decltype(Reversed(cstub)), TReverseImpl<const TStub&>);
    }

    SIMPLE_UNIT_TEST(CopyMoveTest) {
        TState lvalue;
        const TState clvalue;
        UNIT_ASSERT_NO_EXCEPTION(Reversed(lvalue));
        UNIT_ASSERT_NO_EXCEPTION(Reversed(clvalue));
    }
}
