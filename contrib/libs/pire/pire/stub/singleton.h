#pragma once
#include <util/generic/singleton.h>
namespace Pire {
    template<class T>
    const T& DefaultValue() {
	return Default<T>();
    }
}
