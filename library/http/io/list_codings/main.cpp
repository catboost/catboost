#include <library/http/io/stream.h>
#include <util/stream/output.h>

int main() {
    for (auto codec : SupportedCodings()) {
        Cout << codec << Endl;
    }
}
