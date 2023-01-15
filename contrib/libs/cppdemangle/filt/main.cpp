#include <util/stream/input.h>
#include <util/stream/output.h>
#include <util/system/demangle.h>

int main() {
    TString s;
    while (Cin.ReadLine(s)) {
        Cout << CppDemangle(s) << Endl;
    }
}
