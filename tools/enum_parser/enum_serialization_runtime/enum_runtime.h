#pragma once

#include <util/generic/array_ref.h>
#include <util/generic/strbuf.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/generic/serialized_enum.h>

#include <utility>

class IOutputStream;

namespace NEnumSerializationRuntime {
    /// Stores all information about enumeration except its real type
    template <typename TEnumRepresentationType>
    class TEnumDescriptionBase {
    public:
        using TRepresentationType = TEnumRepresentationType;

        struct TEnumStringPair {
            const TRepresentationType Key;
            const TStringBuf Name;
        };

        /// Refers initialization data stored in constexpr-friendly format
        struct TInitializationData {
            const TArrayRef<const TEnumStringPair> NamesInitializer;
            const TArrayRef<const TEnumStringPair> ValuesInitializer;
            const TArrayRef<const TStringBuf> CppNamesInitializer;
            const TStringBuf CppNamesPrefix;
            const TStringBuf ClassName;
        };

    public:
        TEnumDescriptionBase(const TInitializationData& enumInitData);
        ~TEnumDescriptionBase();

        const TString& ToString(TRepresentationType key) const;
        std::pair<bool, TRepresentationType> TryFromString(const TStringBuf name) const;
        TRepresentationType FromString(const TStringBuf name) const;
        void Out(IOutputStream* os, const TRepresentationType key) const;

        const TString& AllEnumNames() const noexcept {
            return AllNames;
        }

        const TVector<TString>& AllEnumCppNames() const noexcept {
            return AllCppNames;
        }

        const TMap<TRepresentationType, TString>& TypelessEnumNames() const noexcept {
            return Names;
        }

        const TVector<TRepresentationType>& TypelessEnumValues() const noexcept {
            return AllValues;
        }

    private:
        TMap<TRepresentationType, TString> Names;
        TMap<TString, TRepresentationType> Values;
        TString AllNames;
        TVector<TString> AllCppNames;
        TString ClassName;
        TVector<TRepresentationType> AllValues;
    };

    /// Wraps TEnumDescriptionBase and performs on-demand casts
    template <typename EEnum, typename TEnumRepresentationType = typename NDetail::TSelectEnumRepresentationType<EEnum>::TType>
    class TEnumDescription: public NDetail::TMappedViewBase<EEnum, TEnumRepresentationType>, private TEnumDescriptionBase<TEnumRepresentationType> {
    public:
        using TBase = TEnumDescriptionBase<TEnumRepresentationType>;
        using TCast = NDetail::TMappedViewBase<EEnum, TEnumRepresentationType>;
        using TBase::AllEnumCppNames;
        using TBase::AllEnumNames;
        using typename TBase::TEnumStringPair;
        using typename TBase::TRepresentationType;
        using typename TBase::TInitializationData;

    public:
        using TBase::TBase;

        const TString& ToString(const EEnum key) const {
            return TBase::ToString(TCast::CastToRepresentationType(key));
        }

        bool FromString(const TStringBuf name, EEnum& ret) const {
            const auto findResult = TBase::TryFromString(name);
            if (findResult.first) {
                ret = TCast::CastFromRepresentationType(findResult.second);
                return true;
            }
            return false;
        }

        EEnum FromString(const TStringBuf name) const {
            return TCast::CastFromRepresentationType(TBase::FromString(name));
        }

        TMappedDictView<EEnum, TString> EnumNames() const noexcept {
            return {TBase::TypelessEnumNames()};
        }

        TMappedArrayView<EEnum> AllEnumValues() const noexcept {
            return {TBase::TypelessEnumValues()};
        }

        void Out(IOutputStream* os, const EEnum key) const {
            TBase::Out(os, TCast::CastToRepresentationType(key));
        }

        static constexpr TEnumStringPair EnumStringPair(const EEnum key, const TStringBuf name) noexcept {
            return {TCast::CastToRepresentationType(key), name};
        }
    };
}
