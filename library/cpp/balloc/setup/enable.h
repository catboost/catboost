#pragma once

namespace NAllocSetup {
    // The IsEnabledByDefault variable should always have static initialization. It is safe to use it in initialization
    // of global and thread-local objects because standard guarantees that static initalization always takes place
    // before dynamic initialization:
    // * C++11 3.6.2.2: "Static initialization shall be performed before any dynamic initialization takes place."
    // * C++17 6.6.2.2: "All static initialization strongly happens before any dynamic initialization."
    // On practice a constant value is just baked into the executable during the linking.
    extern const bool EnableByDefault;
}
