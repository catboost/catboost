#ifndef ERROR_ATTRIBUTES_INL_H_
#error "Direct inclusion of this file is not allowed, include error_attributes.h"
// For the sake of sane code completion.
#include "error_attributes.h"
#endif

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class T>
    requires CConvertibleFromAttributeValue<T>
T TErrorAttributes::Get(TStringBuf key) const
{
    auto value = GetValue(key);
    try {
        return NYT::FromErrorAttributeValue<T>(value);
    } catch (const std::exception& ex) {
        ThrowCannotParseAttributeException(key, ex);
    }
}

template <class T>
    requires CConvertibleFromAttributeValue<T>
typename TOptionalTraits<T>::TOptional TErrorAttributes::Find(TStringBuf key) const
{
    auto value = FindValue(key);
    if (!value) {
        return typename TOptionalTraits<T>::TOptional();
    }
    try {
        return NYT::FromErrorAttributeValue<T>(*value);
    } catch (const std::exception& ex) {
        ThrowCannotParseAttributeException(key, ex);
    }
}

template <class T>
    requires CConvertibleFromAttributeValue<T>
T TErrorAttributes::GetAndRemove(const TKey& key)
{
    auto result = Get<T>(key);
    Remove(key);
    return result;
}

template <class T>
    requires CConvertibleFromAttributeValue<T>
T TErrorAttributes::Get(TStringBuf key, const T& defaultValue) const
{
    return Find<T>(key).value_or(defaultValue);
}

template <class T>
    requires CConvertibleFromAttributeValue<T>
T TErrorAttributes::GetAndRemove(const TKey& key, const T& defaultValue)
{
    if (auto value = Find<T>(key)) {
        Remove(key);
        return *value;
    } else {
        return defaultValue;
    }
}

template <class T>
    requires CConvertibleFromAttributeValue<T>
typename TOptionalTraits<T>::TOptional TErrorAttributes::FindAndRemove(const TKey& key)
{
    auto value = Find<T>(key);
    if (value) {
        Remove(key);
    }
    return value;
}

template <CMergeableDictionary TDictionary>
void TErrorAttributes::MergeFrom(const TDictionary& dict)
{
    for (auto range = AsMergeableRange(dict); const auto& [key, value] : range) {
        SetValue(key, value);
    }
}

////////////////////////////////////////////////////////////////////////////////

namespace NMergeableRangeImpl {

inline TMergeableRange TagInvoke(TTagInvokeTag<AsMergeableRange>, const TErrorAttributes& attributes)
{
    return attributes.ListPairs();
}

} // namespace NMergeableRangeImpl

static_assert(CMergeableDictionary<TErrorAttributes>);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
