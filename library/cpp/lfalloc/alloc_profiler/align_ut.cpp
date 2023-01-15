#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/scope.h>

Y_UNIT_TEST_SUITE(MemAlign) {
    Y_UNIT_TEST(ShouldAlign)
    {
        for (ui64 size = 8; size <= 32 * 1024; size *= 8) {
            for (ui64 align = 8; align <= 4096; align *=2) {
                void* ptr = nullptr;

                int res = posix_memalign(&ptr, align, size);
                UNIT_ASSERT_C(res == 0 && ptr != nullptr, "memalign failed");

                Y_DEFER {
                    free(ptr);
                };

                UNIT_ASSERT_C((uintptr_t)ptr % align == 0, "non aligned memory");
            }
        }
    }
}
