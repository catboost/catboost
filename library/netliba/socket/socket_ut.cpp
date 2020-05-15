#include <library/cpp/unittest/registar.h>
#include "socket.h"
#include <string.h>

Y_UNIT_TEST_SUITE(TestDarwinQuirks) {
#if !defined(_win_)
    Y_UNIT_TEST(TestAlign) {
        for (size_t i = 0; i < 1000; ++i) {
            UNIT_ASSERT_VALUES_EQUAL(Y_CMSG_SPACE(i), CMSG_SPACE(i));
        }
    }
#endif
}

namespace NNetlibaSocket {
    class TNetlibaSocketTosTest: public TTestBase {
        UNIT_TEST_SUITE(TNetlibaSocketTosTest);
        UNIT_TEST(WriteReadTest);
        UNIT_TEST_SUITE_END();

        void WriteReadTest() {
            const ui8 tosValue = 77;
            char tosBuf[TOS_BUFFER_SIZE];
            void* t = CreateTos(tosValue, tosBuf);

            char dataBuf[12];
            const TIoVec iov = CreateIoVec(&dataBuf[0], 12);

            static sockaddr_in6 addr;
            const TMsgHdr hdr = CreateSendMsgHdr(addr, iov, t);

            ui8 readedTos = 0;
            UNIT_ASSERT_EQUAL(true, ReadTos(hdr, &readedTos));
            UNIT_ASSERT_EQUAL(tosValue, readedTos);
        }
    };

    class TNetlibaSocketAuxTest: public TTestBase {
        UNIT_TEST_SUITE(TNetlibaSocketAuxTest);
        UNIT_TEST(MyIPAndTosTest);
        UNIT_TEST_SUITE_END();

        void MyIPAndTosTest() {
            struct sockaddr_in6 myAddr;
            memset(&myAddr, 0, sizeof(myAddr));
            char ctrlBuffer[CTRL_BUFFER_SIZE];
            memset(ctrlBuffer, 0, CTRL_BUFFER_SIZE);
            const ui8 tosValue = 77;

            char dataBuf[12];
            const TIoVec iov = CreateIoVec(&dataBuf[0], 12);

            struct sockaddr_in6 toAddr;
            TMsgHdr hdr = CreateSendMsgHdr(toAddr, iov, ctrlBuffer);
            UNIT_ASSERT_EQUAL(AddSockAuxData(&hdr, tosValue, myAddr, ctrlBuffer, CTRL_BUFFER_SIZE), &hdr);

            {
                struct sockaddr_in6 resAddr;
                ExtractDestinationAddress(hdr, &resAddr);
                UNIT_ASSERT_EQUAL(memcmp(&myAddr.sin6_addr, &resAddr.sin6_addr, sizeof(struct in6_addr)), 0);
                ui8 readedTos = 0;
                UNIT_ASSERT_EQUAL(true, ReadTos(hdr, &readedTos));
                UNIT_ASSERT_EQUAL(tosValue, readedTos);
            }
        }
    };

    UNIT_TEST_SUITE_REGISTRATION(TNetlibaSocketTosTest);
    UNIT_TEST_SUITE_REGISTRATION(TNetlibaSocketAuxTest);
}
