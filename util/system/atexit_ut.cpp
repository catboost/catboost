#include <library/cpp/testing/unittest/registar.h>

#include "atexit.h"
#include <util/generic/singleton.h>

#include <errno.h>

#ifdef _win_
// not implemented
#else
    #include <sys/types.h>
    #include <sys/wait.h>
#endif //_win_

#include <stdio.h>

#ifdef _win_
// not implemented
#else
struct TAtExitParams {
    TAtExitParams(int fd_, const char* str_)
        : fd(fd_)
        , str(str_)
    {
    }

    int fd;
    const char* str;
};

void MyAtExitFunc(void* ptr) {
    THolder<TAtExitParams> params{static_cast<TAtExitParams*>(ptr)};
    if (write(params->fd, params->str, strlen(params->str)) < 0) {
        abort();
    }
}
#endif

class TAtExitTest: public TTestBase {
    UNIT_TEST_SUITE(TAtExitTest);
    UNIT_TEST(TestAtExit)
    UNIT_TEST_SUITE_END();

    void TestAtExit() {
#ifdef _win_
// not implemented
#else
        int ret;
        int pipefd[2];

        ret = pipe(pipefd);
        UNIT_ASSERT(ret == 0);

        pid_t pid = fork();

        if (pid < 0) {
            UNIT_ASSERT(0);
        }

        if (pid > 0) {
            char data[1024];
            int last = 0;

            close(pipefd[1]);

            while (read(pipefd[0], data + last++, 1) > 0 && last < 1024) {
            }
            data[--last] = 0;

            UNIT_ASSERT(strcmp(data, "High prio\nMiddle prio\nLow-middle prio\nLow prio\nVery low prio\n") == 0);
        } else {
            close(pipefd[0]);

            AtExit(MyAtExitFunc, new TAtExitParams(pipefd[1], "Low prio\n"), 3);
            AtExit(MyAtExitFunc, new TAtExitParams(pipefd[1], "Middle prio\n"), 5);
            AtExit(MyAtExitFunc, new TAtExitParams(pipefd[1], "High prio\n"), 7);
            AtExit(MyAtExitFunc, new TAtExitParams(pipefd[1], "Very low prio\n"), 1);
            AtExit(MyAtExitFunc, new TAtExitParams(pipefd[1], "Low-middle prio\n"), 4);

            exit(0);
        }
#endif //_win_
    }
};

UNIT_TEST_SUITE_REGISTRATION(TAtExitTest);

Y_UNIT_TEST_SUITE(TestAtExit) {

    Y_UNIT_TEST(CreateUponDestruction) {
        struct T1 {
        };

        struct T2 {
            ~T2() {
                Singleton<T1>();
            }
        };

        Singleton<T2>();
    }
} // Y_UNIT_TEST_SUITE(TestAtExit)
