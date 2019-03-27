#pragma once

#include <library/containers/flat_hash/lib/map.h>
#include <library/containers/flat_hash/lib/containers.h>
#include <library/containers/flat_hash/lib/probings.h>
#include <library/containers/flat_hash/lib/set.h>
#include <library/containers/flat_hash/lib/size_fitters.h>
#include <library/containers/flat_hash/lib/expanders.h>

#include <util/str_stl.h>

namespace NPrivate {

template <class Key, class T, class Hash, class KeyEqual, class Probing>
using TFlatHashMapImpl = NFlatHash::TMap<Key, T, Hash, KeyEqual,
                                         NFlatHash::TFlatContainer<std::pair<const Key, T>>,
                                         Probing, NFlatHash::TAndSizeFitter,
                                         NFlatHash::TSimpleExpander>;

template <class Key, class T, auto emptyMarker, class Hash, class KeyEqual, class Probing>
using TDenseHashMapImpl =
    NFlatHash::TMap<Key, T, Hash, KeyEqual,
                    NFlatHash::TDenseContainer<std::pair<const Key, T>,
                                               NFlatHash::NMap::TStaticValueMarker<emptyMarker, T>>,
                    Probing, NFlatHash::TAndSizeFitter, NFlatHash::TSimpleExpander>;


template <class T, class Hash, class KeyEqual, class Probing>
using TFlatHashSetImpl = NFlatHash::TSet<T, Hash, KeyEqual,
                                         NFlatHash::TFlatContainer<T>,
                                         Probing, NFlatHash::TAndSizeFitter,
                                         NFlatHash::TSimpleExpander>;

template <class T, auto emptyMarker, class Hash, class KeyEqual, class Probing>
using TDenseHashSetImpl =
    NFlatHash::TSet<T, Hash, KeyEqual,
                    NFlatHash::TDenseContainer<T, NFlatHash::NSet::TStaticValueMarker<emptyMarker>>,
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
          class Probing = NFlatHash::TLinearProbing>
using TFlatHashMap = NPrivate::TFlatHashMapImpl<Key, T, Hash, KeyEqual, Probing>;

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
          class Probing = NFlatHash::TDenseProbing>
using TDenseHashMapStaticMarker = NPrivate::TDenseHashMapImpl<Key, T, emptyMarker,
                                                              Hash, KeyEqual, Probing>;


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
          class Probing = NFlatHash::TLinearProbing>
using TFlatHashSet = NPrivate::TFlatHashSetImpl<T, Hash, KeyEqual, Probing>;

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
          class Probing = NFlatHash::TDenseProbing>
using TDenseHashSetStaticMarker = NPrivate::TDenseHashSetImpl<T, emptyMarker,
                                                              Hash, KeyEqual, Probing>;

}  // namespace NFH
