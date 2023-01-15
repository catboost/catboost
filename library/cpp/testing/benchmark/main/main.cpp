#include <library/cpp/testing/benchmark/bench.h>

#include <util/generic/yexception.h>
#include <util/stream/output.h>

#include <cstdlib>

int main(int argc, char** argv) {
    try {
        return NBench::Main(argc, argv);
    } catch (...) {
        Cerr << CurrentExceptionMessage() << Endl;
    }

    return EXIT_FAILURE;
}
