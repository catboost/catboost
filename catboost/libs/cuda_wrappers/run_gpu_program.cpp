#include "run_gpu_program.h"

void RunGpuProgram(std::function<void()> func) {
    TMaybe<TCatBoostException> message;
    {
        std::thread thread([&]() -> void {
            try {
                func();
            } catch (...) {
                message = TCatBoostException() << CurrentExceptionMessage();
            }
        });
        thread.join();
    }
    if (message) {
        ythrow* message;
    }
}
