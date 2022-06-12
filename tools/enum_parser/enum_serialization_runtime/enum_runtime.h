#pragma once

#include "dispatch_methods.h"
#include "ordered_pairs.h"

#include <util/generic/array_ref.h>
#include <util/generic/strbuf.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/generic/serialized_enum.h>
#include <util/stream/fwd.h>

#include <utility>

namespace NEnumSerializationRuntime {
    /// Stores all information about enumeration except its real type
    template <typename TEnumRepresentationType>
    class TEnumDescriptionBase {
    public:
        using TRepresentationType = TEnumRepresentationType;
        using TEnumStringPair = ::NEnumSerializationRuntime::TEnumStringPair<TRepresentationType>;

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

        TStringBuf ToStringBuf(TRepresentationType key) const;
        static TStringBuf ToStringBufFullScan(const TRepresentationType key, const TInitializationData& enumInitData);
        static TStringBuf ToStringBufSorted(const TRepresentationType key, const TInitializationData& enumInitData);
        static TStringBuf ToStringBufDirect(const TRepresentationType key, const TInitializationData& enumInitData);

        std::pair<bool, TRepresentationType> TryFromString(const TStringBuf name) const;
        static std::pair<bool, TRepresentationType> TryFromStringFullScan(const TStringBuf name, const TInitializationData& enumInitData);
        static std::pair<bool, TRepresentationType> TryFromStringSorted(const TStringBuf name, const TInitializationData& enumInitData);

        TRepresentationType FromString(const TStringBuf name) const;
        static TRepresentationType FromStringFullScan(const TStringBuf name, const TInitializationData& enumInitData);
        static TRepresentationType FromStringSorted(const TStringBuf name, const TInitializationData& enumInitData);

        void Out(IOutputStream* os, const TRepresentationType key) const;
        static void OutFullScan(IOutputStream* os, const TRepresentationType key, const TInitializationData& enumInitData);
        static void OutSorted(IOutputStream* os, const TRepresentationType key, const TInitializationData& enumInitData);
        static void OutDirect(IOutputStream* os, const TRepresentationType key, const TInitializationData& enumInitData);

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
        TMap<TStringBuf, TRepresentationType> Values;
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

    private:
        static bool MapFindResult(std::pair<bool, TEnumRepresentationType> findResult, EEnum& ret) {
            if (findResult.first) {
                ret = TCast::CastFromRepresentationType(findResult.second);
                return true;
            }
            return false;
        }

    public:
        using TBase::TBase;

        // ToString
        // Return reference to singleton preallocated string
        const TString& ToString(const EEnum key) const {
            return TBase::ToString(TCast::CastToRepresentationType(key));
        }

        // ToStringBuf
        TStringBuf ToStringBuf(EEnum key) const {
            return TBase::ToStringBuf(TCast::CastToRepresentationType(key));
        }
        static TStringBuf ToStringBufFullScan(const EEnum key, const TInitializationData& enumInitData) {
            return TBase::ToStringBufFullScan(TCast::CastToRepresentationType(key), enumInitData);
        }
        static TStringBuf ToStringBufSorted(const EEnum key, const TInitializationData& enumInitData) {
            return TBase::ToStringBufSorted(TCast::CastToRepresentationType(key), enumInitData);
        }
        static TStringBuf ToStringBufDirect(const EEnum key, const TInitializationData& enumInitData) {
            return TBase::ToStringBufDirect(TCast::CastToRepresentationType(key), enumInitData);
        }

        // TryFromString-like functons
        // Return false for unknown enumeration names
        bool FromString(const TStringBuf name, EEnum& ret) const {
            return MapFindResult(TBase::TryFromString(name), ret);
        }
        static bool TryFromStringFullScan(const TStringBuf name, EEnum& ret, const TInitializationData& enumInitData) {
            return MapFindResult(TBase::TryFromStringFullScan(name, enumInitData), ret);
        }
        static bool TryFromStringSorted(const TStringBuf name, EEnum& ret, const TInitializationData& enumInitData) {
            return MapFindResult(TBase::TryFromStringSorted(name, enumInitData), ret);
        }

        // FromString
        // Throw exception for unknown enumeration names
        EEnum FromString(const TStringBuf name) const {
            return TCast::CastFromRepresentationType(TBase::FromString(name));
        }
        static EEnum FromStringFullScan(const TStringBuf name, const TInitializationData& enumInitData) {
            return TCast::CastFromRepresentationType(TBase::FromStringFullScan(name, enumInitData));
        }
        static EEnum FromStringSorted(const TStringBuf name, const TInitializationData& enumInitData) {
            return TCast::CastFromRepresentationType(TBase::FromStringSorted(name, enumInitData));
        }

        // Inspection
        TMappedDictView<EEnum, TString> EnumNames() const noexcept {
            return {TBase::TypelessEnumNames()};
        }

        TMappedArrayView<EEnum> AllEnumValues() const noexcept {
            return {TBase::TypelessEnumValues()};
        }

        // Out
        void Out(IOutputStream* os, const EEnum key) const {
            TBase::Out(os, TCast::CastToRepresentationType(key));
        }
        static void OutFullScan(IOutputStream* os, const EEnum key, const TInitializationData& enumInitData) {
            TBase::OutFullScan(os, TCast::CastToRepresentationType(key), enumInitData);
        }
        static void OutSorted(IOutputStream* os, const EEnum key, const TInitializationData& enumInitData) {
            TBase::OutSorted(os, TCast::CastToRepresentationType(key), enumInitData);
        }
        static void OutDirect(IOutputStream* os, const EEnum key, const TInitializationData& enumInitData) {
            TBase::OutDirect(os, TCast::CastToRepresentationType(key), enumInitData);
        }

        static constexpr TEnumStringPair EnumStringPair(const EEnum key, const TStringBuf name) noexcept {
            return {TCast::CastToRepresentationType(key), name};
        }
    };
}
