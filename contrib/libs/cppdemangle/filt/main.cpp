#include <contrib/libs/cppdemangle/demangle.h>

#include <util/stream/input.h>
#include <util/stream/output.h>

int main() {
    TString s;
    THolder<char, TFree> name;
    while (Cin.ReadLine(s)) {
        name = llvm_demangle_gnu3(~s);
        if (name.Get()) {
            Cout << name.Get() << Endl;
        } else {
            Cout << s << Endl;
        }
    }
}
