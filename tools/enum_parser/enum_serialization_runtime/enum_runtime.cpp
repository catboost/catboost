#include "enum_runtime.h"

#include <util/generic/algorithm.h>
#include <util/generic/map.h>
#include <util/generic/yexception.h>
#include <util/stream/output.h>

namespace NEnumSerializationRuntime {
    template <typename TEnumRepresentationType>
    [[noreturn]] static void ThrowUndefinedValueException(const TEnumRepresentationType key, const TStringBuf className) {
        throw yexception() << "Undefined value " << key << " in " << className << ". ";
    }

    template <typename TEnumRepresentationType>
    const TString& TEnumDescriptionBase<TEnumRepresentationType>::ToString(TRepresentationType key) const {
        const auto it = Names.find(key);
        if (Y_LIKELY(it != Names.end())) {
            return it->second;
        }
        ThrowUndefinedValueException(key, ClassName);
    }

    template <typename TEnumRepresentationType>
    std::pair<bool, TEnumRepresentationType> TEnumDescriptionBase<TEnumRepresentationType>::TryFromString(const TStringBuf name) const {
        const auto it = Values.find(name);
        if (it != Values.end()) {
            return {true, it->second};
        }
        return {false, TRepresentationType()};
    }

    template <class TContainer, class TNeedle, class TGetKey>
    static typename TContainer::value_type const* FindPtrInSortedContainer(const TContainer& vec, const TNeedle& needle, TGetKey&& getKey) {
        const auto it = LowerBoundBy(vec.begin(), vec.end(), needle, getKey);
        if (it == vec.end()) {
            return nullptr;
        }
        if (getKey(*it) != needle) {
            return nullptr;
        }
        return std::addressof(*it);
    }

    template <typename TEnumRepresentationType>
    std::pair<bool, TEnumRepresentationType> TEnumDescriptionBase<TEnumRepresentationType>::TryFromStringSorted(const TStringBuf name, const TInitializationData& enumInitData) {
        const auto& vec = enumInitData.ValuesInitializer;
        const auto* ptr = FindPtrInSortedContainer(vec, name, std::mem_fn(&TEnumStringPair::Name));
        if (ptr) {
            return {true, ptr->Key};
        }
        return {false, TRepresentationType()};
    }

    template <typename TEnumRepresentationType>
    std::pair<bool, TEnumRepresentationType> TEnumDescriptionBase<TEnumRepresentationType>::TryFromStringFullScan(const TStringBuf name, const TInitializationData& enumInitData) {
        const auto& vec = enumInitData.ValuesInitializer;
        const auto* ptr = FindIfPtr(vec, [&](const auto& item) { return item.Name == name; });
        if (ptr) {
            return {true, ptr->Key};
        }
        return {false, TRepresentationType()};
    }

    [[noreturn]] static void ThrowUndefinedNameException(const TStringBuf name, const TStringBuf className, const TStringBuf allEnumNames) {
        ythrow yexception() << "Key '" << name << "' not found in enum " << className << ". Valid options are: " << allEnumNames << ". ";
    }

    template <typename TEnumRepresentationType>
    [[noreturn]] static void ThrowUndefinedNameException(const TStringBuf name, const typename TEnumDescriptionBase<TEnumRepresentationType>::TInitializationData& enumInitData) {
        auto exc = __LOCATION__ + yexception() << "Key '" << name << "' not found in enum " << enumInitData.ClassName << ". Valid options are: ";
        const auto& vec = enumInitData.NamesInitializer;
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i != 0) {
                exc << ", ";
            }
            exc << '\'' << vec[i].Name << '\'';
        }
        exc << ". ";
        throw exc;
    }

    template <typename TEnumRepresentationType>
    auto TEnumDescriptionBase<TEnumRepresentationType>::FromString(const TStringBuf name) const -> TRepresentationType {
        const auto findResult = TryFromString(name);
        if (Y_LIKELY(findResult.first)) {
            return findResult.second;
        }
        ThrowUndefinedNameException(name, ClassName, AllEnumNames());
    }

    template <typename TEnumRepresentationType>
    TEnumRepresentationType TEnumDescriptionBase<TEnumRepresentationType>::FromStringFullScan(const TStringBuf name, const TInitializationData& enumInitData) {
        const auto findResult = TryFromStringFullScan(name, enumInitData);
        if (Y_LIKELY(findResult.first)) {
            return findResult.second;
        }
        ThrowUndefinedNameException<TEnumRepresentationType>(name, enumInitData);
    }

    template <typename TEnumRepresentationType>
    TEnumRepresentationType TEnumDescriptionBase<TEnumRepresentationType>::FromStringSorted(const TStringBuf name, const TInitializationData& enumInitData) {
        const auto findResult = TryFromStringSorted(name, enumInitData);
        if (Y_LIKELY(findResult.first)) {
            return findResult.second;
        }
        ThrowUndefinedNameException<TEnumRepresentationType>(name, enumInitData);
    }

    template <typename TEnumRepresentationType>
    TStringBuf TEnumDescriptionBase<TEnumRepresentationType>::ToStringBuf(TRepresentationType key) const {
        return this->ToString(key);
    }

    template <typename TEnumRepresentationType>
    TStringBuf TEnumDescriptionBase<TEnumRepresentationType>::ToStringBufFullScan(const TRepresentationType key, const TInitializationData& enumInitData) {
        const auto& vec = enumInitData.NamesInitializer;
        const auto* ptr = FindIfPtr(vec, [&](const auto& item) { return item.Key == key; });
        if (Y_UNLIKELY(!ptr)) {
            ThrowUndefinedValueException(key, enumInitData.ClassName);
        }
        return ptr->Name;
    }

    template <typename TEnumRepresentationType>
    TStringBuf TEnumDescriptionBase<TEnumRepresentationType>::ToStringBufSorted(const TRepresentationType key, const TInitializationData& enumInitData) {
        const auto& vec = enumInitData.NamesInitializer;
        const auto* ptr = FindPtrInSortedContainer(vec, key, std::mem_fn(&TEnumStringPair::Key));
        if (Y_UNLIKELY(!ptr)) {
            ThrowUndefinedValueException(key, enumInitData.ClassName);
        }
        return ptr->Name;
    }

    template <typename TEnumRepresentationType>
    TStringBuf TEnumDescriptionBase<TEnumRepresentationType>::ToStringBufDirect(const TRepresentationType key, const TInitializationData& enumInitData) {
        const auto& vec = enumInitData.NamesInitializer;
        if (Y_UNLIKELY(vec.empty() || key < vec.front().Key)) {
            ThrowUndefinedValueException(key, enumInitData.ClassName);
        }
        const size_t index = static_cast<size_t>(key - vec.front().Key);
        if (Y_UNLIKELY(index >= vec.size())) {
            ThrowUndefinedValueException(key, enumInitData.ClassName);
        }
        return vec[index].Name;
    }

    template <typename TEnumRepresentationType>
    void TEnumDescriptionBase<TEnumRepresentationType>::Out(IOutputStream* os, const TRepresentationType key) const {
        (*os) << this->ToStringBuf(key);
    }

    template <typename TEnumRepresentationType>
    void TEnumDescriptionBase<TEnumRepresentationType>::OutFullScan(IOutputStream* os, const TRepresentationType key, const TInitializationData& enumInitData) {
        (*os) << ToStringBufFullScan(key, enumInitData);
    }

    template <typename TEnumRepresentationType>
    void TEnumDescriptionBase<TEnumRepresentationType>::OutSorted(IOutputStream* os, const TRepresentationType key, const TInitializationData& enumInitData) {
        (*os) << ToStringBufSorted(key, enumInitData);
    }

    template <typename TEnumRepresentationType>
    void TEnumDescriptionBase<TEnumRepresentationType>::OutDirect(IOutputStream* os, const TRepresentationType key, const TInitializationData& enumInitData) {
        (*os) << ToStringBufDirect(key, enumInitData);
    }

    template <typename TEnumRepresentationType>
    TEnumDescriptionBase<TEnumRepresentationType>::TEnumDescriptionBase(const TInitializationData& enumInitData)
        : ClassName(enumInitData.ClassName)
    {
        const TArrayRef<const TEnumStringPair>& namesInitializer = enumInitData.NamesInitializer;
        const TArrayRef<const TEnumStringPair>& valuesInitializer = enumInitData.ValuesInitializer;
        const TArrayRef<const TStringBuf>& cppNamesInitializer = enumInitData.CppNamesInitializer;

        TMap<TRepresentationType, TString> mapValueToName;
        TMap<TStringBuf, TRepresentationType> mapNameToValue;
        for (const TEnumStringPair& it : namesInitializer) {
            mapValueToName.emplace(it.Key, TString(it.Name));
        }
        for (const TEnumStringPair& it : valuesInitializer) {
            mapNameToValue.emplace(it.Name, it.Key);
        }
        Names = std::move(mapValueToName);
        Values = std::move(mapNameToValue);

        AllValues.reserve(Names.size());
        for (const auto& it : Names) {
            if (!AllNames.empty()) {
                AllNames += ", ";
            }
            AllNames += TString::Join('\'', it.second, '\'');
            AllValues.push_back(it.first);
        }

        AllCppNames.reserve(cppNamesInitializer.size());
        for (const auto& cn : cppNamesInitializer) {
            AllCppNames.push_back(TString::Join(enumInitData.CppNamesPrefix, cn));
        }
    }

    template <typename TEnumRepresentationType>
    TEnumDescriptionBase<TEnumRepresentationType>::~TEnumDescriptionBase() = default;

    // explicit instantiation
    template class TEnumDescriptionBase<int>;
    template class TEnumDescriptionBase<unsigned>;
    template class TEnumDescriptionBase<long long>;
    template class TEnumDescriptionBase<unsigned long long>;
}
