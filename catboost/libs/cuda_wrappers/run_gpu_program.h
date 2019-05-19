#pragma once

#include <catboost/libs/helpers/exception.h>

#include <util/generic/maybe.h>

#include <thread>

//damn acradia, if crash on GPU from main thread => no memcheck results
void RunGpuProgram(std::function<void()> func);
