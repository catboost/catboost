#pragma once

#include "error_attribute.h"
#include "mergeable_dictionary.h"

#include <library/cpp/yt/misc/optional.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

// TODO(arkady-e1ppa): Try switching to TString/std::string eventually.
// representing text-encoded YSON string eventually (maybe).
class TErrorAttributes
{
public:
    using TKey = TErrorAttribute::TKey;
    using TValue = TErrorAttribute::TValue;
    using TKeyValuePair = std::pair<TKey, TValue>;

    //! Returns the list of all keys in the dictionary.
    std::vector<TKey> ListKeys() const;

    //! Returns the list of all key-value pairs in the dictionary.
    std::vector<TKeyValuePair> ListPairs() const;

    //! Sets the value of the attribute.
    void SetAttribute(const TErrorAttribute& attribute);

    //! Removes the attribute.
    //! Returns |true| if the attribute was removed or |false| if there is no attribute with this key.
    bool Remove(const TKey& key);

    //! Removes all attributes.
    void Clear();

    //! Returns |true| iff the given key is present.
    bool Contains(TStringBuf key) const;

    //! Finds the attribute and deserializes its value.
    //! Throws if no such value is found.
    template <class T>
        requires CConvertibleFromAttributeValue<T>
    T Get(TStringBuf key) const;

    //! Same as #Get but removes the value.
    template <class T>
        requires CConvertibleFromAttributeValue<T>
    T GetAndRemove(const TKey& key);

    //! Finds the attribute and deserializes its value.
    //! Uses default value if no such attribute is found.
    template <class T>
        requires CConvertibleFromAttributeValue<T>
    T Get(TStringBuf key, const T& defaultValue) const;

    //! Same as #Get but removes the value if it exists.
    template <class T>
        requires CConvertibleFromAttributeValue<T>
    T GetAndRemove(const TKey& key, const T& defaultValue);

    //! Finds the attribute and deserializes its value.
    //! Returns null if no such attribute is found.
    template <class T>
        requires CConvertibleFromAttributeValue<T>
    typename TOptionalTraits<T>::TOptional Find(TStringBuf key) const;

    //! Same as #Find but removes the value if it exists.
    template <class T>
        requires CConvertibleFromAttributeValue<T>
    typename TOptionalTraits<T>::TOptional FindAndRemove(const TKey& key);

    template <CMergeableDictionary TDictionary>
    void MergeFrom(const TDictionary& dict);

private:
    THashMap<TKey, TValue, THash<TStringBuf>, TEqualTo<TStringBuf>> Map_;

    friend class TErrorOr<void>;
    TErrorAttributes() = default;

    TErrorAttributes(const TErrorAttributes& other) = default;
    TErrorAttributes& operator= (const TErrorAttributes& other) = default;

    TErrorAttributes(TErrorAttributes&& other) = default;
    TErrorAttributes& operator= (TErrorAttributes&& other) = default;

    //! Returns the value of the attribute (null indicates that the attribute is not found).
    std::optional<TValue> FindValue(TStringBuf key) const;

    //! Returns the value of the attribute (throws an exception if the attribute is not found).
    TValue GetValue(TStringBuf key) const;

    //! Sets the value of the attribute.
    void SetValue(const TKey& key, const TValue& value);

    [[noreturn]] static void ThrowCannotParseAttributeException(TStringBuf key, const std::exception& ex);
    [[noreturn]] static void ThrowNoSuchAttributeException(TStringBuf key);
};

bool operator == (const TErrorAttributes& lhs, const TErrorAttributes& rhs);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define ERROR_ATTRIBUTES_INL_H_
#include "error_attributes-inl.h"
#undef ERROR_ATTRIBUTES_INL_H_
