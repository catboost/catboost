#include <openssl/crypto.h>

namespace {
    // Initialize OpenSSL as early as possible
    // in order to prevent any further initializations with different flags.
    //
    // Initialize it with OPENSSL_INIT_NO_ATEXIT thus omitting the cleanup routine at process exit
    // (it looks like it does nothing when openssl is linked statically).
    [[maybe_unused]] auto _ = OPENSSL_init_crypto(OPENSSL_INIT_ENGINE_ALL_BUILTIN | OPENSSL_INIT_NO_ATEXIT, nullptr);
}
