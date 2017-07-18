#include <util/generic/singleton.h>

struct X {
    char Buf[100];
};

char& FF1() noexcept {
    static X x;

    return x.Buf[0];
}

char& FF2() noexcept {
    return Singleton<X>()->Buf[0];
}
