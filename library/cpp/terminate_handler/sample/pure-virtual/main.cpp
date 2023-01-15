
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
    TQux();
    return 0;
}
