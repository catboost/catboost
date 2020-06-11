#include "io_service_impl.h"

#include <library/cpp/unittest/registar.h>

#include <util/stream/str.h>

using namespace NAsio;

Y_UNIT_TEST_SUITE(TIOService) {
    class TTestWriteOperation: public TFdOperation {
    public:
        TTestWriteOperation(PIPEHANDLE fd, size_t& runTimes)
            : TFdOperation(fd, PollWrite)
            , RunTimes(runTimes)
        {
        }

        bool Execute(int ec) override {
            UNIT_ASSERT_VALUES_EQUAL(ec, 0);

#ifdef _win_
            if (::send(Fd(), "", 1, 0) != 1) {
#else
            if (::write(Fd(), "", 1) != 1) {
#endif
                return true;
            }
            return !--RunTimes;
        }

        size_t& RunTimes;
    };

    class TTestReadOperation: public TFdOperation {
    public:
        TTestReadOperation(PIPEHANDLE fd, size_t& runTimes)
            : TFdOperation(fd, PollRead)
            , RunTimes(runTimes)
        {
        }

        bool Execute(int ec) override {
            UNIT_ASSERT_VALUES_EQUAL(ec, 0);

            char byte;
#ifdef _win_
            ssize_t res = ::recv(Fd(), &byte, 1, 0);
#else
            ssize_t res = ::read(Fd(), &byte, 1);
#endif
            if (res != 1) {
                return true;
            }
            return !--RunTimes;
        }

        size_t& RunTimes;
    };

    Y_UNIT_TEST(TTestWriteOp) {
        TIOService::TImpl srv;
        TPipeHandle pipe[2];
        TPipeHandle::Pipe(pipe[0], pipe[1]);

        size_t writeTimes = 5;
        srv.ScheduleOp(new TTestWriteOperation(pipe[1], writeTimes));
        srv.Run(); //<<< 5 times write zero byte and exit

        UNIT_ASSERT_VALUES_EQUAL(writeTimes, 0u);

        size_t readTimes = 5;
        srv.ScheduleOp(new TTestReadOperation(pipe[0], readTimes));
        srv.Run(); //<<< 5 times read byte and exit

        UNIT_ASSERT_VALUES_EQUAL(readTimes, 0u);

        //TODO: more tests
    }
}
