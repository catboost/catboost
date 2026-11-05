#pragma once

#include "flat_hash_map.h"

#include <util/ysaveload.h>

template <class T1, class T2, class T3, class T4, class T5>
class TSerializer<absl::flat_hash_map<T1, T2, T3, T4, T5>>: public TMapSerializer<absl::flat_hash_map<T1, T2, T3, T4, T5>, false> {};
