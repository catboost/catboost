#include <library/cpp/unittest/utmain.h>
#include <library/cpp/terminate_handler/terminate_handler.h>

int main(int argc, char** argv) {
    SetFancyTerminateHandler();
    return NUnitTest::RunMain(argc, argv);
}
