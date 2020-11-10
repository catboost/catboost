#include "hook.h"

namespace {
    NTesting::THook* BeforeInitHead = nullptr;
    NTesting::THook* BeforeRunHead = nullptr;
    NTesting::THook* AfterRunHead = nullptr;

    void RegisterHook(NTesting::THook*& head, NTesting::THook* hook) {
        hook->Next = head;
        head = hook;
    }

    void CallHooks(NTesting::THook* head) {
        while (nullptr != head) {
            if (nullptr != head->Fn) {
                (*head->Fn)();
            }
            head = head->Next;
        }
    }
}

void NTesting::THook::RegisterBeforeInit(NTesting::THook* hook) noexcept {
    RegisterHook(BeforeInitHead, hook);
}

void NTesting::THook::CallBeforeInit() {
    CallHooks(BeforeInitHead);
}

void NTesting::THook::RegisterBeforeRun(NTesting::THook* hook) noexcept {
    RegisterHook(BeforeRunHead, hook);
}

void NTesting::THook::CallBeforeRun() {
    CallHooks(BeforeRunHead);
}

void NTesting::THook::RegisterAfterRun(NTesting::THook* hook) noexcept {
    RegisterHook(AfterRunHead, hook);
}

void NTesting::THook::CallAfterRun() {
    CallHooks(AfterRunHead);
}
