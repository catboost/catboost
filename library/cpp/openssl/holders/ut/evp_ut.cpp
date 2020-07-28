#include "evp.h"

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(Evp) {
    Y_UNIT_TEST(Cipher) {
        NOpenSSL::TEvpCipherCtx ctx;
        UNIT_ASSERT(ctx);
    }

    Y_UNIT_TEST(Md) {
        NOpenSSL::TEvpMdCtx ctx;
        UNIT_ASSERT(ctx);
    }
}
