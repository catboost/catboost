#include <util/generic/yexception.h>

#include "../../terminate_handler.h"

void Foo(unsigned i = 0) {
    if (i >= 10) {
        ythrow yexception() << "from Foo()";
    } else {
        Foo(i + 1);
    }
}

int main() {
    SetFancyTerminateHandler();
    Foo();
    return 0;
}
