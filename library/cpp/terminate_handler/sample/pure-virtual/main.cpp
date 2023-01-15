#include "../../terminate_handler.h"

struct TFoo {
    TFoo() {
        Baz();
    }

    void Baz() {
        Bar();
    }

    virtual void Bar() = 0;
};

struct TQux: public TFoo {
    void Bar() override {
    }
};

int main() {
    SetFancyTerminateHandler();
    TQux();
    return 0;
}
