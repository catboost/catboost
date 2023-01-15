#pragma once

#include <util/generic/maybe.h>

#include <thread>

//damn acradia, if crash on GPU from main thread => no memcheck results
void RunGpuProgram(std::function<void()> func);
