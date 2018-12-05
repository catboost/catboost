#pragma once

#include <util/digest/city.h>
#include <util/generic/vector.h>

template <class T>
inline ui64 VecCityHash(const TVector<T>& data) {
    return CityHash64(reinterpret_cast<const char*>(data.data()), sizeof(T) * data.size());
}
