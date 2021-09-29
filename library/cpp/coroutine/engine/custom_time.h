#pragma once

#include <util/datetime/base.h>

namespace NCoro {
class ITime {
  public:
    virtual TInstant Now() = 0;
};
}
