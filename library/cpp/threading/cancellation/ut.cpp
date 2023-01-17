#include "cancellation_token.h"

#include <library/cpp/testing/unittest/registar.h>

using namespace NThreading;

Y_UNIT_TEST_SUITE(Cancellation) {
    Y_UNIT_TEST(IsCancellationRequested) {
        TCancellationTokenSource source;
        auto const token = source.Token();
        UNIT_ASSERT(!source.IsCancellationRequested());
        UNIT_ASSERT(!token.IsCancellationRequested());
        source.Cancel();
        UNIT_ASSERT(source.IsCancellationRequested());
        UNIT_ASSERT(token.IsCancellationRequested());
    }

    Y_UNIT_TEST(ThrowIfCancellationRequested) {
        TCancellationTokenSource source;
        auto const token = source.Token();
        UNIT_ASSERT_NO_EXCEPTION(token.ThrowIfCancellationRequested());
        source.Cancel();
        UNIT_ASSERT_EXCEPTION(token.ThrowIfCancellationRequested(), TOperationCancelledException);
    }

    Y_UNIT_TEST(Wait) {
        TCancellationTokenSource source;
        auto const token = source.Token();
        UNIT_ASSERT(!token.Wait(TDuration::MilliSeconds(10)));
        source.Cancel();
        UNIT_ASSERT(token.Wait(TDuration::MilliSeconds(10)));
    }

    Y_UNIT_TEST(Future) {
        TCancellationTokenSource source;
        auto const future = source.Token().Future();
        UNIT_ASSERT(!future.HasValue());
        UNIT_ASSERT(!future.HasException());
        source.Cancel();
        UNIT_ASSERT(future.HasValue());
    }

    Y_UNIT_TEST(Default) {
        auto const& token = TCancellationToken::Default();
        UNIT_ASSERT(!token.IsCancellationRequested());
    }
}
