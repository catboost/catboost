#include "enum_runtime.h"

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

    [[noreturn]] static void ThrowUndefinedNameException(const TStringBuf name, const TStringBuf className, const TStringBuf allEnumNames) {
        ythrow yexception() << "Key '" << name << "' not found in enum " << className << ". Valid options are: " << allEnumNames << ". ";
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
    void TEnumDescriptionBase<TEnumRepresentationType>::Out(IOutputStream* os, const TRepresentationType key) const {
        (*os) << this->ToString(key);
    }

    template <typename TEnumRepresentationType>
    TEnumDescriptionBase<TEnumRepresentationType>::TEnumDescriptionBase(const TInitializationData& enumInitData)
        : ClassName(enumInitData.ClassName)
    {
        const TArrayRef<const TEnumStringPair>& namesInitializer = enumInitData.NamesInitializer;
        const TArrayRef<const TEnumStringPair>& valuesInitializer = enumInitData.ValuesInitializer;
        const TArrayRef<const TStringBuf>& cppNamesInitializer = enumInitData.CppNamesInitializer;

        TMap<TRepresentationType, TString> mapValueToName;
        TMap<TString, TRepresentationType> mapNameToValue;
        const bool bijectiveHint = (namesInitializer.data() == valuesInitializer.data() && namesInitializer.size() == valuesInitializer.size());
        if (bijectiveHint) {
            for (const TEnumStringPair& it : namesInitializer) {
                TString name{it.Name};
                mapValueToName.emplace(it.Key, name);
                mapNameToValue.emplace(std::move(name), it.Key);
            }
        } else {
            for (const TEnumStringPair& it : namesInitializer) {
                mapValueToName.emplace(it.Key, TString(it.Name));
            }
            for (const TEnumStringPair& it : valuesInitializer) {
                mapNameToValue.emplace(TString(it.Name), it.Key);
            }
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
