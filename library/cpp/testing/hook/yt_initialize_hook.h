#pragma once

// Since we need to call NYT::Initialize immediately on starting the program, we separate it from other BEFORE_INIT hooks.
// This weak function is overriden in C++ client (see yt/cpp/mapreduce/tests/yt_initialize_hook/yt_hook.cpp).
__attribute__((weak)) void InitializeYt(int, char**) {
}
