#pragma once

namespace NTesting {
    /**
     * Hook class and registration system.
     *
     * Default implementation of the `main` function for G_BENCHMARK and GTEST calls these hooks when executing.
     * This is a useful feature if you want to customize behaviour of the `main` function,
     * but you don't want to write `main` yourself.
     *
     * Hooks form an intrusive linked list that's built at application startup. Note that hooks execute
     * in arbitrary order.
     *
     * Use macros below to define hooks.
     */
    struct THook {
        using TFn = void (*)();

        TFn Fn = nullptr;
        THook* Next = nullptr;

        static void RegisterBeforeInit(THook* hook) noexcept;

        static void CallBeforeInit();

        struct TRegisterBeforeInit {
            explicit TRegisterBeforeInit(THook* hook) noexcept {
                THook::RegisterBeforeInit(hook);
            }
        };

        static void RegisterBeforeRun(THook* hook) noexcept;

        static void CallBeforeRun();

        struct TRegisterBeforeRun {
            explicit TRegisterBeforeRun(THook* hook) noexcept {
                THook::RegisterBeforeRun(hook);
            }
        };

        static void RegisterAfterRun(THook* hook) noexcept;

        static void CallAfterRun();

        struct TRegisterAfterRun {
            explicit TRegisterAfterRun(THook* hook) noexcept {
                THook::RegisterAfterRun(hook);
            }
        };
    };

    /**
     * Called right before initializing test programm
     *
     * This hook is intended for setting up default parameters. If you're doing initialization, consider
     * using `Y_TEST_HOOK_BEFORE_RUN` instead.
     *
     * *Note:* hooks execute in arbitrary order.
     *
     *
     * # Usage
     *
     * Instantiate this class in a cpp file. Pass a unique name for your hook,
     * implement body right after macro instantiation:
     *
     * ```
     * Y_TEST_HOOK_BEFORE_INIT(SetupParams) {
     *     // hook code
     * }
     * ```
     */
#define Y_TEST_HOOK_BEFORE_INIT(N)                                                 \
        void N();                                                                  \
        ::NTesting::THook N##Hook{&N, nullptr};                                    \
        ::NTesting::THook::TRegisterBeforeInit N##HookReg{&N##Hook};               \
        void N()

    /**
     * Called right before launching tests.
     *
     * Hooks execute in arbitrary order. As such, we recommend using this hook to set up an event listener,
     * and performing initialization and cleanup in the corresponding event handlers. This is better than performing
     * initialization and cleanup directly in the hook's code because it gives more control over
     * order in which initialization is performed.
     *
     *
     * # Usage
     *
     * Instantiate this class in a cpp file. Pass a unique name for your hook,
     * implement body right after macro instantiation:
     *
     * ```
     * Y_TEST_HOOK_BEFORE_RUN(InitMyApp) {
     *     // hook code
     * }
     * ```
     */
#define Y_TEST_HOOK_BEFORE_RUN(N)                                                  \
        void N();                                                                  \
        ::NTesting::THook N##Hook{&N, nullptr};                                    \
        ::NTesting::THook::TRegisterBeforeRun N##HookReg{&N##Hook};                \
        void N()

    /**
     * Called after all tests has finished, just before program exit.
     *
     * This hook is intended for simple cleanup routines that don't care about order in which hooks are executed.
     * For more complex cases, we recommend using `Y_TEST_HOOK_BEFORE_RUN`.
     *
     *
     * # Usage
     *
     * Instantiate this class in a cpp file. Pass a unique name for your hook,
     * implement body right after macro instantiation:
     *
     * ```
     * Y_TEST_HOOK_AFTER_RUN(StopMyApp) {
     *     // hook code
     * }
     * ```
     */
#define Y_TEST_HOOK_AFTER_RUN(N)                                             \
        void N();                                                            \
        ::NTesting::THook N##Hook{&N, nullptr};                              \
        ::NTesting::THook::TRegisterAfterRun N##HookReg{&N##Hook};           \
        void N()
}
