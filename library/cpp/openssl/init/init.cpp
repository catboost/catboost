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
            OPENSSL_init_crypto(OPENSSL_INIT_NO_ATEXIT, nullptr);
        }

        inline ~TInitSsl() {
            OPENSSL_cleanup();
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
    (void)SingletonWithPriority<TInitSsl, 0>();
}
