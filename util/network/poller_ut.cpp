#include <library/cpp/testing/unittest/registar.h>
#include <util/system/error.h>

#include "pair.h"
#include "poller.h"
#include "pollerimpl.h"

Y_UNIT_TEST_SUITE(TSocketPollerTest) {
    Y_UNIT_TEST(TestSimple) {
        SOCKET sockets[2];
        UNIT_ASSERT(SocketPair(sockets) == 0);

        TSocketHolder s1(sockets[0]);
        TSocketHolder s2(sockets[1]);

        TSocketPoller poller;
        poller.WaitRead(sockets[1], (void*)17);

        UNIT_ASSERT_VALUES_EQUAL(nullptr, poller.WaitT(TDuration::Zero()));
        UNIT_ASSERT_VALUES_EQUAL(nullptr, poller.WaitT(TDuration::Zero()));

        for (ui32 i = 0; i < 3; ++i) {
            char buf[] = {18};
            UNIT_ASSERT_VALUES_EQUAL(1, send(sockets[0], buf, 1, 0));

            UNIT_ASSERT_VALUES_EQUAL((void*)17, poller.WaitT(TDuration::Zero()));

            UNIT_ASSERT_VALUES_EQUAL(1, recv(sockets[1], buf, 1, 0));
            UNIT_ASSERT_VALUES_EQUAL(18, buf[0]);

            UNIT_ASSERT_VALUES_EQUAL(nullptr, poller.WaitT(TDuration::Zero()));
            UNIT_ASSERT_VALUES_EQUAL(nullptr, poller.WaitT(TDuration::Zero()));
        }
    }

    Y_UNIT_TEST(TestSimpleOneShot) {
        SOCKET sockets[2];
        UNIT_ASSERT(SocketPair(sockets) == 0);

        TSocketHolder s1(sockets[0]);
        TSocketHolder s2(sockets[1]);

        TSocketPoller poller;

        UNIT_ASSERT_VALUES_EQUAL(nullptr, poller.WaitT(TDuration::Zero()));
        UNIT_ASSERT_VALUES_EQUAL(nullptr, poller.WaitT(TDuration::Zero()));

        for (ui32 i = 0; i < 3; ++i) {
            poller.WaitReadOneShot(sockets[1], (void*)17);

            char buf[1];

            buf[0] = i + 20;

            UNIT_ASSERT_VALUES_EQUAL(1, send(sockets[0], buf, 1, 0));

            UNIT_ASSERT_VALUES_EQUAL((void*)17, poller.WaitT(TDuration::Zero()));

            UNIT_ASSERT_VALUES_EQUAL(1, recv(sockets[1], buf, 1, 0));
            UNIT_ASSERT_VALUES_EQUAL(char(i + 20), buf[0]);

            UNIT_ASSERT_VALUES_EQUAL(nullptr, poller.WaitT(TDuration::Zero()));
            UNIT_ASSERT_VALUES_EQUAL(nullptr, poller.WaitT(TDuration::Zero()));

            buf[0] = i + 21;

            UNIT_ASSERT_VALUES_EQUAL(1, send(sockets[0], buf, 1, 0));

            // this fails if socket is not oneshot
            UNIT_ASSERT_VALUES_EQUAL(nullptr, poller.WaitT(TDuration::Zero()));
            UNIT_ASSERT_VALUES_EQUAL(nullptr, poller.WaitT(TDuration::Zero()));

            UNIT_ASSERT_VALUES_EQUAL(1, recv(sockets[1], buf, 1, 0));
            UNIT_ASSERT_VALUES_EQUAL(char(i + 21), buf[0]);
        }
    }

    Y_UNIT_TEST(TestItIsSafeToUnregisterUnregisteredDescriptor) {
        SOCKET sockets[2];
        UNIT_ASSERT(SocketPair(sockets) == 0);

        TSocketHolder s1(sockets[0]);
        TSocketHolder s2(sockets[1]);

        TSocketPoller poller;

        poller.Unwait(s1);
    }

    Y_UNIT_TEST(TestItIsSafeToReregisterDescriptor) {
        SOCKET sockets[2];
        UNIT_ASSERT(SocketPair(sockets) == 0);

        TSocketHolder s1(sockets[0]);
        TSocketHolder s2(sockets[1]);

        TSocketPoller poller;

        poller.WaitRead(s1, nullptr);
        poller.WaitRead(s1, nullptr);
        poller.WaitWrite(s1, nullptr);
    }

    Y_UNIT_TEST(TestSimpleEdgeTriggered) {
        SOCKET sockets[2];
        UNIT_ASSERT(SocketPair(sockets) == 0);

        TSocketHolder s1(sockets[0]);
        TSocketHolder s2(sockets[1]);

        SetNonBlock(sockets[1]);

        TSocketPoller poller;

        UNIT_ASSERT_VALUES_EQUAL(nullptr, poller.WaitT(TDuration::Zero()));
        UNIT_ASSERT_VALUES_EQUAL(nullptr, poller.WaitT(TDuration::Zero()));

        for (ui32 i = 0; i < 3; ++i) {
            poller.WaitReadWriteEdgeTriggered(sockets[1], (void*)17);

            // notify about writeble
            UNIT_ASSERT_VALUES_EQUAL((void*)17, poller.WaitT(TDuration::Zero()));
            UNIT_ASSERT_VALUES_EQUAL(nullptr, poller.WaitT(TDuration::Zero()));

            char buf[2];

            buf[0] = i + 10;
            buf[1] = i + 20;

            // send one byte
            UNIT_ASSERT_VALUES_EQUAL(1, send(sockets[0], buf, 1, 0));

            UNIT_ASSERT_VALUES_EQUAL((void*)17, poller.WaitT(TDuration::Zero()));
            UNIT_ASSERT_VALUES_EQUAL(nullptr, poller.WaitT(TDuration::Zero()));

            // restart without reading
            poller.RestartReadWriteEdgeTriggered(sockets[1], (void*)17, false);

            // after restart read and write might generate separate events
            {
                void* events[3];
                size_t count = poller.WaitT(events, 3, TDuration::Zero());
                UNIT_ASSERT_GE(count, 1);
                UNIT_ASSERT_LE(count, 2);
                UNIT_ASSERT_VALUES_EQUAL(events[0], (void*)17);
            }

            UNIT_ASSERT_VALUES_EQUAL(nullptr, poller.WaitT(TDuration::Zero()));

            // second two more bytes
            UNIT_ASSERT_VALUES_EQUAL(2, send(sockets[0], buf, 2, 0));

            // here poller could notify or not because we haven't seen end
            Y_UNUSED(poller.WaitT(TDuration::Zero()));

            // recv one, leave two
            UNIT_ASSERT_VALUES_EQUAL(1, recv(sockets[1], buf, 1, 0));
            UNIT_ASSERT_VALUES_EQUAL(char(i + 10), buf[0]);

            // nothing new
            UNIT_ASSERT_VALUES_EQUAL(nullptr, poller.WaitT(TDuration::Zero()));
            UNIT_ASSERT_VALUES_EQUAL(nullptr, poller.WaitT(TDuration::Zero()));

            // recv the rest
            UNIT_ASSERT_VALUES_EQUAL(2, recv(sockets[1], buf, 2, 0));
            UNIT_ASSERT_VALUES_EQUAL(char(i + 10), buf[0]);
            UNIT_ASSERT_VALUES_EQUAL(char(i + 20), buf[1]);

            // still nothing new
            UNIT_ASSERT_VALUES_EQUAL(nullptr, poller.WaitT(TDuration::Zero()));
            UNIT_ASSERT_VALUES_EQUAL(nullptr, poller.WaitT(TDuration::Zero()));

            // hit end
            ClearLastSystemError();
            UNIT_ASSERT_VALUES_EQUAL(-1, recv(sockets[1], buf, 1, 0));
            UNIT_ASSERT_VALUES_EQUAL(EAGAIN, LastSystemError());

            // restart after end (noop for epoll)
            poller.RestartReadWriteEdgeTriggered(sockets[1], (void*)17, true);

            // send and recv byte
            UNIT_ASSERT_VALUES_EQUAL(1, send(sockets[0], buf, 1, 0));

            UNIT_ASSERT_VALUES_EQUAL((void*)17, poller.WaitT(TDuration::Zero()));
            UNIT_ASSERT_VALUES_EQUAL(nullptr, poller.WaitT(TDuration::Zero()));

            // recv and see end
            UNIT_ASSERT_VALUES_EQUAL(1, recv(sockets[1], buf, 2, 0));
            UNIT_ASSERT_VALUES_EQUAL(char(i + 10), buf[0]);

            UNIT_ASSERT_VALUES_EQUAL(nullptr, poller.WaitT(TDuration::Zero()));
            UNIT_ASSERT_VALUES_EQUAL(nullptr, poller.WaitT(TDuration::Zero()));

            // the same but send before restart
            UNIT_ASSERT_VALUES_EQUAL(1, send(sockets[0], buf, 1, 0));

            // restart after end (noop for epoll)
            poller.RestartReadWriteEdgeTriggered(sockets[1], (void*)17, true);

            UNIT_ASSERT_VALUES_EQUAL((void*)17, poller.WaitT(TDuration::Zero()));
            UNIT_ASSERT_VALUES_EQUAL(nullptr, poller.WaitT(TDuration::Zero()));

            UNIT_ASSERT_VALUES_EQUAL(1, recv(sockets[1], buf, 2, 0));
            UNIT_ASSERT_VALUES_EQUAL(char(i + 10), buf[0]);

            UNIT_ASSERT_VALUES_EQUAL(nullptr, poller.WaitT(TDuration::Zero()));
            UNIT_ASSERT_VALUES_EQUAL(nullptr, poller.WaitT(TDuration::Zero()));

            poller.Unwait(sockets[1]);
        }
    }

#if defined(HAVE_EPOLL_POLLER)
    Y_UNIT_TEST(TestRdhup) {
        SOCKET sockets[2];
        UNIT_ASSERT(SocketPair(sockets) == 0);

        TSocketHolder s1(sockets[0]);
        TSocketHolder s2(sockets[1]);

        char buf[1] = {0};
        UNIT_ASSERT_VALUES_EQUAL(1, send(s1, buf, 1, 0));
        shutdown(s1, SHUT_WR);

        using TPoller = TGenericPoller<TEpollPoller<TWithoutLocking>>;
        TPoller poller;
        poller.Set((void*)17, s2, CONT_POLL_RDHUP);

        TPoller::TEvent e;
        UNIT_ASSERT_VALUES_EQUAL(poller.WaitD(&e, 1, TDuration::Zero().ToDeadLine()), 1);
        UNIT_ASSERT_EQUAL(TPoller::ExtractStatus(&e), 0);
        UNIT_ASSERT_EQUAL(TPoller::ExtractFilter(&e), CONT_POLL_RDHUP);
        UNIT_ASSERT_EQUAL(TPoller::ExtractEvent(&e), (void*)17);
    }

    Y_UNIT_TEST(TestSetSocketErrors) {
        TGenericPoller<TEpollPoller<TWithoutLocking>> poller;

        UNIT_ASSERT_EXCEPTION_CONTAINS(poller.Set(nullptr, Max<int>(), CONT_POLL_READ), TSystemError, "epoll add failed");
        UNIT_ASSERT_EXCEPTION_CONTAINS(poller.Set(nullptr, Max<int>(), CONT_POLL_READ | CONT_POLL_MODIFY), TSystemError, "epoll modify failed");
    }
#endif
} // Y_UNIT_TEST_SUITE(TSocketPollerTest)
