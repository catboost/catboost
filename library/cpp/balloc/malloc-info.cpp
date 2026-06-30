#include <library/cpp/malloc/api/malloc.h>

#include <string.h>

using namespace NMalloc;

extern "C" void DisableBalloc();
extern "C" void EnableBalloc();
extern "C" bool BallocDisabled();

namespace {
    bool SetAllocParam(const char* name, const char* value) {
        if (strcmp(name, "disable") == 0) {
            if (value == nullptr || strcmp(value, "false") != 0) {
                // all values other than "false" are considred to be "true" for compatibility
                DisableBalloc();
            } else {
                EnableBalloc();
            }
            return true;
        }
        return false;
    }

    bool CheckAllocParam(const char* name, bool defaultValue) {
        if (strcmp(name, "disable") == 0) {
            return BallocDisabled();
        }
        return defaultValue;
    }
}

TMallocInfo NMalloc::MallocInfo() {
    TMallocInfo r;

    r.Name = "balloc";
    r.SetParam = SetAllocParam;
    r.CheckParam = CheckAllocParam;

    return r;
}
