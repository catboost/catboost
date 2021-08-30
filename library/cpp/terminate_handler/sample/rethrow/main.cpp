#include <util/generic/yexception.h>


void Bar() {
    ythrow yexception() << "from Foo()";
}

void Foo() {
    try {
        Bar();
    } catch (...) {
        Cerr << "caught; rethrowing\n";
        throw;
    }
}

int main() {
    Foo();
    return 0;
}
