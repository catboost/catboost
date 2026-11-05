#include <library/cpp/testing/common/env.h>

#include <library/cpp/testing/gtest/gtest.h>

#include <dlfcn.h>

#include <atomic>
#include <string>
#include <thread>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

constexpr int StripeCount = 64;

TEST(TAccessSpreaderDynamicTlsTest, LateLoadedSymbolicDsoSupportsPreExistingWorkerAcrossUnload)
{
    using TGetStripe = int (*)();
    using TGetInitialized = bool (*)();

    std::atomic<bool> workerReady = false;
    std::atomic<bool> startWorker = false;
    std::atomic<bool> workerPaused = false;
    std::atomic<bool> finishWorker = false;
    TGetStripe getStripe = nullptr;
    int workerStripe = 0;
    std::thread worker([&] {
        workerReady.store(true, std::memory_order::release);
        while (!startWorker.load(std::memory_order::acquire)) {
            std::this_thread::yield();
        }
        if (!getStripe) {
            return;
        }
        workerStripe = getStripe();
        workerPaused.store(true, std::memory_order::release);
        while (!finishWorker.load(std::memory_order::acquire)) {
            std::this_thread::yield();
        }
    });
    while (!workerReady.load(std::memory_order::acquire)) {
        std::this_thread::yield();
    }

    auto libraryPath = BinaryPath(
        "library/cpp/yt/system/unittests/access_spreader_dynamic_tls/native_symbolic/libaccess_spreader_native_symbolic.so");
    auto* library = dlopen(libraryPath.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!library) {
        startWorker.store(true, std::memory_order::release);
        worker.join();
        FAIL() << dlerror();
    }

    getStripe = reinterpret_cast<TGetStripe>(
        dlsym(library, "GetCurrentAccessStripeFromNativeLibrary"));
    auto getConstructorStripe = reinterpret_cast<TGetStripe>(
        dlsym(library, "GetConstructorStripe"));
    auto getConstructorInitialized = reinterpret_cast<TGetInitialized>(
        dlsym(library, "GetConstructorInitialized"));
    if (!getStripe || !getConstructorStripe || !getConstructorInitialized) {
        auto* error = dlerror();
        std::string errorMessage = error ? error : "Failed to resolve access spreader test symbols";
        getStripe = nullptr;
        startWorker.store(true, std::memory_order::release);
        worker.join();
        dlclose(library);
        FAIL() << errorMessage;
    }

    EXPECT_TRUE(getConstructorInitialized());
    EXPECT_LT(getConstructorStripe(), StripeCount);
    EXPECT_LT(getStripe(), StripeCount);

    startWorker.store(true, std::memory_order::release);
    while (!workerPaused.load(std::memory_order::acquire)) {
        std::this_thread::yield();
    }

    int closeResult = dlclose(library);
    finishWorker.store(true, std::memory_order::release);
    worker.join();

    EXPECT_EQ(closeResult, 0);
    EXPECT_LT(workerStripe, StripeCount);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
