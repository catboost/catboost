#pragma once

#include <library/cpp/yt/misc/enum.h>
#include <library/cpp/yt/misc/port.h>
#include <library/cpp/yt/misc/static_initializer.h>

#include <library/cpp/yt/string/format.h>

#include <util/generic/hash_set.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

class TErrorCodeRegistry
{
public:
    static TErrorCodeRegistry* Get();

    struct TErrorCodeInfo
    {
        std::string Namespace;
        //! Human-readable error code name.
        std::string Name;

        bool operator==(const TErrorCodeInfo& rhs) const;
    };

    struct TErrorCodeRangeInfo
    {
        int From;
        int To;
        std::string Namespace;
        std::function<std::string(int code)> Formatter;

        TErrorCodeInfo Get(int code) const;
        bool Intersects(const TErrorCodeRangeInfo& other) const;
        bool Contains(int value) const;
    };

    //! Retrieves info from registered codes and code ranges.
    TErrorCodeInfo Get(int code) const;

    //! Retrieves information about registered codes.
    THashMap<int, TErrorCodeInfo> GetAllErrorCodes() const;

    //! Retrieves information about registered code ranges.
    std::vector<TErrorCodeRangeInfo> GetAllErrorCodeRanges() const;

    //! Registers a single error code.
    void RegisterErrorCode(int code, const TErrorCodeInfo& errorCodeInfo);

    //! Registers a range of error codes given a human-readable code to name formatter.
    void RegisterErrorCodeRange(int from, int to, std::string namespaceName, std::function<std::string(int code)> formatter);

    static std::string ParseNamespace(const std::type_info& errorCodeEnumTypeInfo);

private:
    THashMap<int, TErrorCodeInfo> CodeToInfo_;
    std::vector<TErrorCodeRangeInfo> ErrorCodeRanges_;

    void CheckCodesAgainstRanges() const;
};

void FormatValue(
    TStringBuilderBase* builder,
    const TErrorCodeRegistry::TErrorCodeInfo& errorCodeInfo,
    TStringBuf spec);

void FormatValue(
    TStringBuilderBase* builder,
    const TErrorCodeRegistry::TErrorCodeRangeInfo& errorCodeInfo,
    TStringBuf spec);

////////////////////////////////////////////////////////////////////////////////

#define YT_DEFINE_ERROR_ENUM(seq) \
    DEFINE_ENUM(EErrorCode, seq); \
    YT_ATTRIBUTE_USED inline const void* ErrorEnum_EErrorCode = [] { \
        for (auto errorCode : ::NYT::TEnumTraits<EErrorCode>::GetDomainValues()) { \
            ::NYT::TErrorCodeRegistry::Get()->RegisterErrorCode( \
                static_cast<int>(errorCode), \
                {::NYT::TErrorCodeRegistry::ParseNamespace(typeid(EErrorCode)), ToString(errorCode)}); \
        } \
        return nullptr; \
    } ()

////////////////////////////////////////////////////////////////////////////////

//! NB: This macro should only by used in cpp files.
#define YT_DEFINE_ERROR_CODE_RANGE(from, to, namespaceName, formatter) \
    YT_STATIC_INITIALIZER( \
        ::NYT::TErrorCodeRegistry::Get()->RegisterErrorCodeRange( \
            from, \
            to, \
            namespaceName, \
            formatter));

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
