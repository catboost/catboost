#include <util/generic/yexception.h>


void Foo(unsigned i = 0) {
    if (i >= 10) {
        ythrow yexception() << "from Foo()";
    } else {
        Foo(i + 1);
    }
}

int main() {
    Foo();
    return 0;
}
