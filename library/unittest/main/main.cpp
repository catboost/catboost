#include <library/unittest/utmain.h>
#include <library/terminate_handler/terminate_handler.h>

int main(int argc, char** argv) {
    SetFancyTerminateHandler();
    return NUnitTest::RunMain(argc, argv);
}
