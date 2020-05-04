#include "../../segv_handler.h"

void Bar(int* x) {
    *x = 11;
}

void Foo(int* x) {
    Bar(x);
}

int main() {
    InstallSegvHandler();
    Foo((int*)1);
    return 0;
}
