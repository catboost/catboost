#pragma once

#include "flat_hash_set.h"

#include <util/ysaveload.h>

template <class T1, class T2, class T3, class T4>
class TSerializer<absl::flat_hash_set<T1, T2, T3, T4>>: public TSetSerializer<absl::flat_hash_set<T1, T2, T3, T4>, false> {};
