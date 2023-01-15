#include <library/cpp/unittest/registar.h>

#include "pair.h"
#include "poller.h"

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

            UNIT_ASSERT_VALUES_EQUAL((void*)17, poller.WaitT(TDuration::Zero()));
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
            UNIT_ASSERT_VALUES_EQUAL(-1, recv(sockets[1], buf, 1, 0));
            UNIT_ASSERT_VALUES_EQUAL(EAGAIN, errno);

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
}
