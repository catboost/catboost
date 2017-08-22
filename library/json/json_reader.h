#pragma once

#include "json_value.h"

#include <library/json/common/defs.h>
#include <library/json/fast_sax/parser.h>

#include <util/generic/yexception.h>

#include <util/stream/input.h>
#include <util/stream/str.h>
#include <util/stream/mem.h>


namespace NJson {

struct TJsonReaderConfig {
    TJsonReaderConfig();

    // js-style comments (both // and /**/)
    bool AllowComments = false;
    bool DontValidateUtf8 = false;

    void SetBufferSize(size_t bufferSize);
    size_t GetBufferSize() const;

private:
    size_t BufferSize;
};

bool ReadJsonTree(TStringBuf in, TJsonValue *out, bool throwOnError = false);
bool ReadJsonTree(TStringBuf in, bool allowComments, TJsonValue *out, bool throwOnError = false);
bool ReadJsonTree(TStringBuf in, const TJsonReaderConfig *config, TJsonValue *out, bool throwOnError = false);

bool ReadJsonTree(IInputStream *in, TJsonValue *out, bool throwOnError = false);
bool ReadJsonTree(IInputStream *in, bool allowComments, TJsonValue *out, bool throwOnError = false);
bool ReadJsonTree(IInputStream *in, const TJsonReaderConfig *config, TJsonValue *out, bool throwOnError = false);

bool ReadJson(IInputStream *in, TJsonCallbacks *callbacks);
bool ReadJson(IInputStream *in, bool allowComments, TJsonCallbacks *callbacks);
bool ReadJson(IInputStream *in, const TJsonReaderConfig *config, TJsonCallbacks *callbacks);


inline bool ValidateJson(IInputStream* in, const TJsonReaderConfig* config, bool throwOnError = false) {
    TJsonCallbacks c(throwOnError);
    return ReadJson(in, config, &c);
}

inline bool ValidateJson(TStringBuf in, const TJsonReaderConfig& config = TJsonReaderConfig(), bool throwOnError = false) {
    TMemoryInput min(~in, +in);
    return ValidateJson(&min, &config, throwOnError);
}

inline bool ValidateJsonThrow(IInputStream* in, const TJsonReaderConfig* config) {
    return ValidateJson(in, config, true);
}

inline bool ValidateJsonThrow(TStringBuf in, const TJsonReaderConfig& config = TJsonReaderConfig()) {
    return ValidateJson(in, config, true);
}

class TParserCallbacks: public TJsonCallbacks {
public:
    TParserCallbacks(TJsonValue &value, bool throwOnError = false);
    bool OnNull() override;
    bool OnBoolean(bool val) override;
    bool OnInteger(long long val) override;
    bool OnUInteger(unsigned long long val) override;
    bool OnString(const TStringBuf &val) override;
    bool OnDouble(double val) override;
    bool OnOpenArray() override;
    bool OnCloseArray() override;
    bool OnOpenMap() override;
    bool OnCloseMap() override;
    bool OnMapKey(const TStringBuf &val) override;
protected:
    TJsonValue &Value;
    TString Key;
    yvector<TJsonValue *> ValuesStack;

    enum {
        START,
        AFTER_MAP_KEY,
        IN_MAP,
        IN_ARRAY,
        FINISH
    } CurrentState;

    template <class T> bool SetValue(const T &value) {
        switch(CurrentState) {
        case START:
            Value.SetValue(value);
            break;
        case AFTER_MAP_KEY:
            ValuesStack.back()->InsertValue(Key, value);
            CurrentState = IN_MAP;
            break;
        case IN_ARRAY:
            ValuesStack.back()->AppendValue(value);
            break;
        case IN_MAP:
        case FINISH:
            return false;
        default:
            ythrow yexception() << "TParserCallbacks::SetValue invalid enum";
        }
        return true;
    }

    bool OpenComplexValue(EJsonValueType type);
    bool CloseComplexValue();
};

//// relaxed json, used in library/scheme
bool ReadJsonFastTree(TStringBuf in, TJsonValue* out, bool throwOnError = false);

}
