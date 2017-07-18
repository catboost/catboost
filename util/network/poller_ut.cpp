#include <library/unittest/registar.h>

#include "pair.h"
#include "poller.h"

SIMPLE_UNIT_TEST_SUITE(TSocketPollerTest) {
    SIMPLE_UNIT_TEST(TestSimple) {
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

    SIMPLE_UNIT_TEST(TestSimpleOneShot) {
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

    SIMPLE_UNIT_TEST(TestItIsSafeToUnregisterUnregisteredDescriptor) {
        SOCKET sockets[2];
        UNIT_ASSERT(SocketPair(sockets) == 0);

        TSocketHolder s1(sockets[0]);
        TSocketHolder s2(sockets[1]);

        TSocketPoller poller;

        poller.Unwait(s1);
    }

    SIMPLE_UNIT_TEST(TestItIsSafeToReregisterDescriptor) {
        SOCKET sockets[2];
        UNIT_ASSERT(SocketPair(sockets) == 0);

        TSocketHolder s1(sockets[0]);
        TSocketHolder s2(sockets[1]);

        TSocketPoller poller;

        poller.WaitRead(s1, nullptr);
        poller.WaitRead(s1, nullptr);
        poller.WaitWrite(s1, nullptr);
    }
}
