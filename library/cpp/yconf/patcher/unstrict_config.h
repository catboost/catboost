#pragma once

#include <library/cpp/charset/ci_string.h>
#include <library/cpp/json/json_value.h>
#include <library/cpp/yconf/conf.h>

#include <util/string/vector.h>

class TUnstrictConfig
   : public TYandexConfig {
public:
    const Section* GetSection(const TString& path);
    bool AddSection(const TString& path);
    TString GetValue(const TString& path) const;
    //return true if value was changed;
    bool SetValue(const TString& path, const TString& value);
    bool Remove(const TString& path);
    bool RemoveAll(const TString& path);
    bool PatchEntry(const TString& path, const TString& value, const TString& prefix = "");

    [[nodiscard]] bool ParseJson(const NJson::TJsonValue& json);

    TString ToString() const;
    NJson::TJsonValue ToJson() const;

public:
    static void ToJsonPatch(const Section& section, NJson::TJsonValue& result, const TString& preffix);
    static void ToJson(const Section& section, NJson::TJsonValue& result);
    static void ToJson(const TYandexConfig& config, NJson::TJsonValue& result);
    static void ToJson(const TString& section, NJson::TJsonValue& result);

    template <class T>
    static NJson::TJsonValue ToJson(const T& entity) {
        NJson::TJsonValue result;
        ToJson(entity, result);
        return result;
    }

protected:
    bool OnBeginSection(Section& sec) override;

private:
    struct TPathUnit {
        TCiString Name;
        size_t BeginIndex;
        size_t EndIndex;

        inline TPathUnit(const TCiString& name, size_t beginIndex, size_t endIndex)
            : Name(name)
            , BeginIndex(beginIndex)
            , EndIndex(endIndex)
        {
            Y_ABORT_UNLESS(EndIndex >= BeginIndex);
        }
        inline TPathUnit(const TCiString& name, size_t index)
            : TPathUnit(name, index, index)
        {
        }
        inline TPathUnit(const TCiString& name)
            : TPathUnit(name, 0)
        {
        }
    };

    using TPathIterator = TVector<TString>::const_iterator;

private:
    bool ParseJson(const NJson::TJsonValue& json, const TString& path);
    TPathUnit ProcessPathUnit(const TString& element) const;
    const Section* GetSection(const Section* section, const TVector<TString>::const_iterator& begin, const TVector<TString>::const_iterator& end) const;
    Section* GetSection(Section* section, const TVector<TString>::const_iterator& begin, const TVector<TString>::const_iterator& end, bool add);
    bool RemoveAllSections(Section& parent, const TString& name);
    bool RemoveSection(Section& parent, const TString& name);
    bool RemoveDirective(Section& parent, const TString& name);

    TVector<const Section*> GetSections(const Section* section, TPathIterator begin, TPathIterator end) const;
    TVector<Section*> GetSections(Section* section, TPathIterator begin, TPathIterator end, bool add);

private:
    TVector<TString> Strings;
};

void SectionToStream(const TYandexConfig::Section* section, IOutputStream& stream, ui16 level = 0);
