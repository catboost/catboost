#include "init.h"

#include <util/generic/singleton.h>
#include <util/generic/vector.h>
#include <util/generic/ptr.h>
#include <util/generic/buffer.h>

#include <util/system/yassert.h>
#include <util/system/mutex.h>
#include <util/system/thread.h>

#include <util/random/entropy.h>
#include <util/stream/input.h>

#include <openssl/bio.h>
#include <openssl/ssl.h>
#include <openssl/err.h>
#include <openssl/rand.h>
#include <openssl/conf.h>
#include <openssl/crypto.h>

namespace {
    struct TInitSsl {
        struct TOpensslLocks {
            inline TOpensslLocks()
                : Mutexes(CRYPTO_num_locks())
            {
                for (auto& mpref : Mutexes) {
                    mpref.Reset(new TMutex());
                }
            }

            inline void LockOP(int mode, int n) {
                auto& mutex = *Mutexes.at(n);

                if (mode & CRYPTO_LOCK) {
                    mutex.Acquire();
                } else {
                    mutex.Release();
                }
            }

            TVector<TAutoPtr<TMutex>> Mutexes;
        };

        inline TInitSsl() {
#if OPENSSL_VERSION_NUMBER >= 0x10101020L // 1.1.1b
            OPENSSL_init_crypto(OPENSSL_INIT_NO_ATEXIT, nullptr);
#elif OPENSSL_VERSION_NUMBER >= 0x10100000L
            OPENSSL_init_crypto(OPENSSL_INIT_LOAD_CONFIG, nullptr);
#else
            SSL_library_init();
            OPENSSL_config(nullptr);
            SSL_load_error_strings();
            OpenSSL_add_all_algorithms();
            ERR_load_BIO_strings();
            CRYPTO_set_id_callback(ThreadIdFunction);
            CRYPTO_set_locking_callback(LockingFunction);
#endif 

#if OPENSSL_VERSION_NUMBER < 0x10101000L
            do {
                char buf[128];
                EntropyPool().Load(buf, sizeof(buf));
                RAND_seed(buf, sizeof(buf));
            } while (!RAND_status());
#endif
        }

        inline ~TInitSsl() {
#if OPENSSL_VERSION_NUMBER >= 0x10101020L // 1.1.1b
            OPENSSL_cleanup();
#elif OPENSSL_VERSION_NUMBER < 0x10100000L
            CRYPTO_set_id_callback(nullptr);
            CRYPTO_set_locking_callback(nullptr);
            ERR_free_strings();
            EVP_cleanup();
#endif
        }

        static void LockingFunction(int mode, int n, const char* /*file*/, int /*line*/) {
            Singleton<TOpensslLocks>()->LockOP(mode, n);
        }

        static unsigned long ThreadIdFunction() {
            return TThread::CurrentThreadId();
        }
    };
}

void InitOpenSSL() {
#if OPENSSL_VERSION_NUMBER >= 0x10101020L // 1.1.1b
    (void)SingletonWithPriority<TInitSsl, 0>();
#else
    (void)Singleton<TInitSsl>();
#endif
}
