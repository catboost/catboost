#pragma once

#include "cow_string.h"

#include <util/ysaveload.h>

template <>
class TSerializer<TCowString>: public TVectorSerializer<TCowString> {
};
