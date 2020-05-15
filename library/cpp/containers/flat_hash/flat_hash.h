#pragma once

#include <library/cpp/containers/flat_hash/lib/map.h>
#include <library/cpp/containers/flat_hash/lib/containers.h>
#include <library/cpp/containers/flat_hash/lib/probings.h>
#include <library/cpp/containers/flat_hash/lib/set.h>
#include <library/cpp/containers/flat_hash/lib/size_fitters.h>
#include <library/cpp/containers/flat_hash/lib/expanders.h>

#include <util/str_stl.h>

namespace NPrivate {

template <class Key, class T, class Hash, class KeyEqual, class Probing, class Alloc>
using TFlatHashMapImpl = NFlatHash::TMap<Key, T, Hash, KeyEqual,
                                         NFlatHash::TFlatContainer<std::pair<const Key, T>, Alloc>,
                                         Probing, NFlatHash::TAndSizeFitter,
                                         NFlatHash::TSimpleExpander>;

template <class Key, class T, auto emptyMarker, class Hash, class KeyEqual, class Probing, class Alloc>
using TDenseHashMapImpl =
    NFlatHash::TMap<Key, T, Hash, KeyEqual,
                    NFlatHash::TDenseContainer<std::pair<const Key, T>,
                                               NFlatHash::NMap::TStaticValueMarker<emptyMarker, T>,
                                               Alloc>,
                    Probing, NFlatHash::TAndSizeFitter, NFlatHash::TSimpleExpander>;


template <class T, class Hash, class KeyEqual, class Probing, class Alloc>
using TFlatHashSetImpl = NFlatHash::TSet<T, Hash, KeyEqual,
                                         NFlatHash::TFlatContainer<T, Alloc>,
                                         Probing, NFlatHash::TAndSizeFitter,
                                         NFlatHash::TSimpleExpander>;

template <class T, auto emptyMarker, class Hash, class KeyEqual, class Probing, class Alloc>
using TDenseHashSetImpl =
    NFlatHash::TSet<T, Hash, KeyEqual,
                    NFlatHash::TDenseContainer<T, NFlatHash::NSet::TStaticValueMarker<emptyMarker>, Alloc>,
                    Probing, NFlatHash::TAndSizeFitter, NFlatHash::TSimpleExpander>;

}  // namespace NPrivate

namespace NFH {

/* flat_map: Fast and highly customizable hash map.
 *
 * Most features would be available soon.
 * Until that time we strongly insist on using only class aliases listed below.
 */

/* Simpliest open addressing hash map.
 * Uses additional array to denote status of every bucket.
 * Default probing is linear.
 * Currently available probings:
 * * TLinearProbing
 * * TQuadraticProbing
 * * TDenseProbing
 */
template <class Key,
          class T,
          class Hash = THash<Key>,
          class KeyEqual = std::equal_to<>,
          class Probing = NFlatHash::TLinearProbing,
          class Alloc = std::allocator<std::pair<const Key, T>>>
using TFlatHashMap = NPrivate::TFlatHashMapImpl<Key, T, Hash, KeyEqual, Probing, Alloc>;

/* Open addressing table with user specified marker for empty buckets.
 * Currently available probings:
 * * TLinearProbing
 * * TQuadraticProbing
 * * TDenseProbing
 */
template <class Key,
          class T,
          auto emptyMarker,
          class Hash = THash<Key>,
          class KeyEqual = std::equal_to<>,
          class Probing = NFlatHash::TDenseProbing,
          class Alloc = std::allocator<std::pair<const Key, T>>>
using TDenseHashMapStaticMarker = NPrivate::TDenseHashMapImpl<Key, T, emptyMarker,
                                                              Hash, KeyEqual, Probing, Alloc>;


/* flat_set: Fast and highly customizable hash set.
 *
 * Most features would be available soon.
 * Until that time we strongly insist on using only class aliases listed below.
 */

/* Simpliest open addressing hash map.
 * Uses additional array to denote status of every bucket.
 * Default probing is linear.
 * Currently available probings:
 * * TLinearProbing
 * * TQuadraticProbing
 * * TDenseProbing
 */
template <class T,
          class Hash = THash<T>,
          class KeyEqual = std::equal_to<>,
          class Probing = NFlatHash::TLinearProbing,
          class Alloc = std::allocator<T>>
using TFlatHashSet = NPrivate::TFlatHashSetImpl<T, Hash, KeyEqual, Probing, Alloc>;

/* Open addressing table with user specified marker for empty buckets.
 * Currently available probings:
 * * TLinearProbing
 * * TQuadraticProbing
 * * TDenseProbing
 */
template <class T,
          auto emptyMarker,
          class Hash = THash<T>,
          class KeyEqual = std::equal_to<>,
          class Probing = NFlatHash::TDenseProbing,
          class Alloc = std::allocator<T>>
using TDenseHashSetStaticMarker = NPrivate::TDenseHashSetImpl<T, emptyMarker,
                                                              Hash, KeyEqual, Probing, Alloc>;

}  // namespace NFH
