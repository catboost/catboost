#include <library/cpp/openssl/method/io.h>

#include <library/cpp/testing/unittest/registar.h>

class TTestIO : public NOpenSSL::TAbstractIO {
public:
    int Write(const char* data, size_t dlen, size_t* written) override {
        Y_UNUSED(data);
        *written = dlen;
        return 1;
    }

    int Read(char* data, size_t dlen, size_t* readbytes) override {
        Y_UNUSED(data);
        Y_UNUSED(dlen);
        *readbytes = 0;
        return 0;
    }

    int Puts(const char* buf) override {
        if (buf == nullptr) {
            return 0;
        }

        return strlen(buf);
    }

    int Gets(char* buf, int size) override {
        Y_UNUSED(buf);
        Y_UNUSED(size);
        return 0;
    }

    void Flush() override {

    }
};

Y_UNIT_TEST_SUITE(IO) {
    Y_UNIT_TEST(AbstractIO) {
        static const char s[] = "12345";

        TTestIO test;

        UNIT_ASSERT(BIO_write(test, s, sizeof(s)) == sizeof(s));
        UNIT_ASSERT(BIO_puts(test, s) == strlen(s));

        char buf[128];
        UNIT_ASSERT(BIO_read(test, buf, sizeof(buf)) == 0);
        UNIT_ASSERT(BIO_gets(test, buf, sizeof(buf)) == 0);
    }
}
