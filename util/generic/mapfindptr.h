#pragma once

#include <type_traits>
#include <utility>

/** MapFindPtr usage:

if (T* value = MapFindPtr(myMap, someKey) {
    Cout << *value;
}

*/

template <class Map, class K>
inline auto MapFindPtr(Map& map, const K& key) {
    auto i = map.find(key);

    return (i == map.end() ? nullptr : &i->second);
}

template <class Map, class K>
inline auto MapFindPtr(const Map& map, const K& key) {
    auto i = map.find(key);

    return (i == map.end() ? nullptr : &i->second);
}

/** helper for THashMap/TMap */
template <class Derived>
struct TMapOps {
    template <class K>
    inline auto FindPtr(const K& key) {
        return MapFindPtr(static_cast<Derived&>(*this), key);
    }

    template <class K>
    inline auto FindPtr(const K& key) const {
        return MapFindPtr(static_cast<const Derived&>(*this), key);
    }

    template <class K, class DefaultValue>
    inline auto Value(const K& key, DefaultValue&& defaultValue) const {
        auto found = FindPtr(key);
        return found ? *found : std::forward<DefaultValue>(defaultValue);
    }

    template <class K, class V>
    inline const V& ValueRef(const K& key, V& defaultValue) const {
        static_assert(std::is_same<std::remove_const_t<V>, typename Derived::mapped_type>::value, "Passed default value must have the same type as the underlying map's mapped_type.");

        if (auto found = FindPtr(key)) {
            return *found;
        }
        return defaultValue;
    }

    template <class K, class V>
    inline const V& ValueRef(const K& key, V&& defaultValue) const = delete;
};
