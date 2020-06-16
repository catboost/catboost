#include <library/cpp/testing/unittest/registar.h>
#include <library/cpp/testing/unittest/tests_data.h>

bool IsFreePort(ui16 port) {
    TInet6StreamSocket sock;
    TSockAddrInet6 addr("::", port);
    Y_ENSURE(SetSockOpt(sock, SOL_SOCKET, SO_REUSEADDR, 1) == 0);
    SetReuseAddressAndPort(sock);
    if (sock.Bind(&addr) == 0) {
        return true;
    }
    return false;
}

void get_port_ranges() {
    for (int i = 1; i < 10; ++i) {
        TPortManager pm;
        ui16 port = pm.GetPortsRange(1024, i);
        for (int p = port; p < port + i; ++p) {
            UNIT_ASSERT(IsFreePort(p));
        }
    }
}

Y_UNIT_TEST_SUITE(TestTPortManager) {
    Y_UNIT_TEST(ParallelRun0) {get_port_ranges();}
    Y_UNIT_TEST(ParallelRun1) {get_port_ranges();}
    Y_UNIT_TEST(ParallelRun2) {get_port_ranges();}
    Y_UNIT_TEST(ParallelRun3) {get_port_ranges();}
    Y_UNIT_TEST(ParallelRun4) {get_port_ranges();}
    Y_UNIT_TEST(ParallelRun5) {get_port_ranges();}
    Y_UNIT_TEST(ParallelRun6) {get_port_ranges();}
    Y_UNIT_TEST(ParallelRun7) {get_port_ranges();}
    Y_UNIT_TEST(ParallelRun8) {get_port_ranges();}
    Y_UNIT_TEST(ParallelRun9) {get_port_ranges();}
}
