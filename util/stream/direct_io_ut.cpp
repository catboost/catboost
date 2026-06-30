#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/string.h>
#include <util/generic/array_size.h>

#include "buffered.h"
#include "direct_io.h"

Y_UNIT_TEST_SUITE(TDirectIOTests) {
    // Decrease numBufToWrite further if tests continue to time out
    static void Y_NO_INLINE Test(EOpenMode mode, size_t numBufToWrite) {
        const char TEMPLATE[] = "qwertyuiopQWERTYUIOPasdfghjklASD";
        const auto TEMPLATE_SIZE = Y_ARRAY_SIZE(TEMPLATE) - 1;
        static_assert(TEMPLATE_SIZE > 0, "must be greater than zero");

        const size_t BUFFER_SIZE = 32 * 1024;
        static_assert(0 == BUFFER_SIZE % TEMPLATE_SIZE, "must be divisible");

        const size_t CHUNK_SIZE_TO_READ = 512;
        static_assert(0 == CHUNK_SIZE_TO_READ % TEMPLATE_SIZE, "must be divisible");

        // filling buffer
        // TEMPLATE|TEMPLATE|TEMPLATE|...
        auto&& buffer = TBuffer{BUFFER_SIZE};
        for (size_t i = 0; i < BUFFER_SIZE / TEMPLATE_SIZE; ++i) {
            buffer.Append(TEMPLATE, TEMPLATE_SIZE);
        }

        // filling file
        // TEMPLATE|TEMPLATE|TEMPLATE|...
        const auto fileName = TString("test.file");
        auto&& directIOBuffer = TDirectIOBufferedFile{fileName, RdWr | CreateAlways | mode};
        {
            auto&& output = TRandomAccessFileOutput{directIOBuffer};
            for (size_t i = 0; i < numBufToWrite; ++i) {
                output.Write(buffer.Data(), BUFFER_SIZE);
            }
        }

        auto&& reader = TRandomAccessFileInput{directIOBuffer, 0};
        auto&& input = TBufferedInput{&reader, 1 << 17};
        auto bytesRead = size_t{};
        while (auto len = input.Read(buffer.Data(), CHUNK_SIZE_TO_READ)) {
            bytesRead += len;
            while (len) {
                if (len < TEMPLATE_SIZE) {
                    UNIT_ASSERT(!memcmp(buffer.Data(), TEMPLATE, len));
                    len = 0;
                } else {
                    UNIT_ASSERT(!memcmp(buffer.Data(), TEMPLATE, TEMPLATE_SIZE));
                    len -= TEMPLATE_SIZE;
                }
            }
        }

        UNIT_ASSERT_VALUES_EQUAL(bytesRead, numBufToWrite * BUFFER_SIZE);
    }

    Y_UNIT_TEST(ReadWriteTest) {
        Test(0, 100 * 32);
    }

    Y_UNIT_TEST(ReadWriteDirectTest) {
        Test(Direct, 100 * 4);
    }

    Y_UNIT_TEST(ReadWriteDirectSeqTest) {
        Test(Direct | Seq, 100 * 4);
    }
} // Y_UNIT_TEST_SUITE(TDirectIOTests)
