#include "zerocopy_output.h"

#include <library/unittest/registar.h>

// This version of memory output stream is written here only
// for testing IZeroCopyOutput implementation of DoWrite.
class TSimpleMemoryOutput: public IZeroCopyOutput {
public:
    TSimpleMemoryOutput(void* buf, size_t len) noexcept
        : Buf_(static_cast<char*>(buf))
        , End_(Buf_ + len)
    {
    }

private:
    size_t DoNext(void** ptr) override {
        Y_ENSURE(Buf_ < End_, AsStringBuf("memory output stream exhausted"));
        *ptr = Buf_;
        return End_ - Buf_;
    }

    void DoAdvance(size_t len) override {
        char* end = Buf_ + len;
        Y_ENSURE(end <= End_, AsStringBuf("memory output stream exhausted"));
        Buf_ = end;
    }

    char* Buf_;
    char* End_;
};

Y_UNIT_TEST_SUITE(TestZerocopyOutput) {
    Y_UNIT_TEST(Write) {
        char buffer[20];
        TSimpleMemoryOutput output(buffer, sizeof(buffer));
        output << "1"
               << "22"
               << "333"
               << "4444"
               << "55555";

        const char* const result = "1"
                                   "22"
                                   "333"
                                   "4444"
                                   "55555";
        UNIT_ASSERT(0 == memcmp(buffer, result, strlen(result)));
    }
}
