#pragma once
// HIP compatibility shim: redirect a <cub/...> include to hipCUB.
#include <hipcub/hipcub.hpp>
#ifndef cub
#define cub hipcub
#endif
