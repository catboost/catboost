#include <util/generic/yexception.h>

#include "../../terminate_handler.h"

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
    SetFancyTerminateHandler();
    Foo();
    return 0;
}
