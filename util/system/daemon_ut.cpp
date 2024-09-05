#include "daemon.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/network/pair.h>
#include <util/network/socket.h>
#include <util/system/pipe.h>

Y_UNIT_TEST_SUITE(TDaemonTest) {
#ifdef _unix_
    template <typename Func>
    static bool ProcessBuffer(Func&& func, void* bufin, size_t size) {
        char* buf = (char*)bufin;
        do {
            const ssize_t bytesDone = func(buf, size);
            if (bytesDone == 0) {
                return false;
            }

            if (bytesDone < 0) {
                if (errno == EAGAIN || errno == EINTR) {
                    continue;
                } else {
                    return false;
                }
            }

            buf += bytesDone;
            size -= bytesDone;
        } while (size != 0);

        return true;
    }

    const int size = 1024 * 4;
    const int pagesSize = sizeof(int) * size;

    Y_UNIT_TEST(WaitForMessageSocket) {
        using namespace NDaemonMaker;
        SOCKET sockets[2];
        SocketPair(sockets, false, true);
        TSocket sender(sockets[0]);
        TSocket receiver(sockets[1]);

        int status = -1;
        int* pages = new int[size];

        memset(pages, 0, pagesSize);
        if (MakeMeDaemon(closeStdIoOnly, openDevNull, chdirNone, returnFromParent)) {
            sender.Close();
            UNIT_ASSERT(ProcessBuffer([&receiver](char* ptr, size_t sz) -> size_t { return receiver.Recv(ptr, sz); }, &status, sizeof(status)));
            UNIT_ASSERT(ProcessBuffer([&receiver](char* ptr, size_t sz) -> size_t { return receiver.Recv(ptr, sz); }, pages, pagesSize));
            UNIT_ASSERT(memchr(pages, 0, pagesSize) == nullptr);
        } else {
            receiver.Close();
            status = 0;
            UNIT_ASSERT(ProcessBuffer([&sender](char* ptr, size_t sz) -> size_t { return sender.Send(ptr, sz); }, &status, sizeof(status)));
            memset(pages, 1, pagesSize);
            UNIT_ASSERT(ProcessBuffer([&sender](char* ptr, size_t sz) -> size_t { return sender.Send(ptr, sz); }, pages, pagesSize));
            exit(0);
        }
        UNIT_ASSERT(status == 0);

        delete[] pages;
    }

    Y_UNIT_TEST(WaitForMessagePipe) {
        using namespace NDaemonMaker;
        TPipeHandle sender;
        TPipeHandle receiver;
        TPipeHandle::Pipe(receiver, sender);

        int status = -1;
        int* pages = new int[size];
        memset(pages, 0, pagesSize);
        if (MakeMeDaemon(closeStdIoOnly, openDevNull, chdirNone, returnFromParent)) {
            sender.Close();
            UNIT_ASSERT(ProcessBuffer([&receiver](char* ptr, size_t sz) -> size_t { return receiver.Read(ptr, sz); }, &status, sizeof(status)));
            UNIT_ASSERT(ProcessBuffer([&receiver](char* ptr, size_t sz) -> size_t { return receiver.Read(ptr, sz); }, pages, pagesSize));
            UNIT_ASSERT(memchr(pages, 0, pagesSize) == nullptr);
        } else {
            receiver.Close();
            status = 0;
            UNIT_ASSERT(ProcessBuffer([&sender](char* ptr, size_t sz) -> size_t { return sender.Write(ptr, sz); }, &status, sizeof(status)));
            memset(pages, 1, pagesSize);
            UNIT_ASSERT(ProcessBuffer([&sender](char* ptr, size_t sz) -> size_t { return sender.Write(ptr, sz); }, pages, pagesSize));
            exit(0);
        }
        UNIT_ASSERT(status == 0);

        delete[] pages;
    }
#endif
} // Y_UNIT_TEST_SUITE(TDaemonTest)
