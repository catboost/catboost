#pragma once

#include <functional>

//damn acradia, if crash on GPU from main thread => no memcheck results
void RunGpuProgram(std::function<void()> func);
