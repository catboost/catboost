#include "config.h"
#include "markup.h"
#include "ini.h"

#include <library/cpp/archive/yarchive.h>

#include <library/cpp/json/json_reader.h>
#include <library/cpp/json/json_writer.h>

#include <library/cpp/lua/eval.h>

#include <util/string/cast.h>
#include <util/string/strip.h>
#include <util/string/type.h>

#include <util/stream/output.h>
#include <util/stream/file.h>
#include <util/memory/blob.h>

#include <util/generic/singleton.h>
#include <util/stream/str.h>

using namespace NConfig;
using namespace NJson;

namespace {
    const unsigned char CODE[] = {
#include "code.inc"
    };

    struct TCode: public TArchiveReader {
        inline TCode()
            : TArchiveReader(TBlob::NoCopy(CODE, sizeof(CODE)))
        {
        }

        static inline TCode& Instance() {
            return *Singleton<TCode>();
        }
    };

    class TPreprocessor: public IInputStream {
    public:
        inline TPreprocessor(const TGlobals& g, IInputStream* in)
            : I_(nullptr, 0)
            , S_(in)
        {
            E_.SetVars(g);
        }

        size_t DoRead(void* ptr, size_t len) override {
            while (true) {
                const size_t read = I_.Read(ptr, len);

                if (read) {
                    return read;
                }

                do {
                    if (!S_.ReadLine(C_)) {
                        return 0;
                    }
                } while (IsComment(C_));

                C_ = E_.Preprocess(C_);
                C_.append('\n');
                I_.Reset(C_.data(), C_.size());
            }
        }

        static inline bool IsComment(TStringBuf s) {
            s = StripString(s);

            return !s.empty() && s[0] == '#';
        }

    private:
        TMemoryInput I_;
        TBufferedInput S_;
        TLuaEval E_;
        TString C_;
    };
}

void TConfig::DumpJson(IOutputStream& out) const {
    TString tmp;

    {
        TStringOutput out2(tmp);

        ToJson(out2);
    }

    {
        TJsonValue v;
        TStringInput in(tmp);

        ReadJsonTree(&in, &v);
        WriteJson(&out, &v, true, true);
    }
}

TConfig TConfig::FromJson(IInputStream& in, const TGlobals& g) {
    class TJSONReader: public TJsonCallbacks {
    public:
        inline TJSONReader()
            : Cur(nullptr)
        {
        }

        inline bool OnBoolean(bool b) override {
            *Next() = ConstructValue(b);

            return true;
        }

        inline bool OnInteger(long long i) override {
            *Next() = ConstructValue((i64)i);

            return true;
        }

        inline bool OnUInteger(unsigned long long i) override {
            *Next() = ConstructValue((ui64)i);

            return true;
        }

        inline bool OnDouble(double d) override {
            *Next() = ConstructValue(d);

            return true;
        }

        inline bool OnString(const TStringBuf& s) override {
            *Next() = ConstructValue(ToString(s));

            return true;
        }

        inline bool OnOpenMap() override {
            Push(ConstructValue(TDict()));

            return true;
        }

        inline bool OnCloseMap() override {
            Pop();

            return true;
        }

        inline bool OnOpenArray() override {
            Push(ConstructValue(TArray()));

            return true;
        }

        inline bool OnCloseArray() override {
            Pop();

            return true;
        }

        inline bool OnMapKey(const TStringBuf& k) override {
            if (S.empty()) {
                ythrow yexception() << "shit happen";
            }

            Cur = &S.back().GetNonConstant<TDict>()[ToString(k)];

            return true;
        }

        inline void Push(const TConfig& el) {
            *Next() = el;
            S.push_back(el);
        }

        inline void Pop() {
            if (S.empty()) {
                ythrow yexception() << "shit happen";
            }

            S.pop_back();
        }

        inline TConfig* Next() {
            if (S.empty()) {
                return &Root;
            }

            TConfig& top = S.back();

            if (top.IsA<TArray>()) {
                TArray& arr = top.GetNonConstant<TArray>();

                arr.emplace_back();

                return &arr.back();
            }

            if (top.IsA<TDict>()) {
                if (Cur) {
                    TConfig* ret = Cur;

                    Cur = nullptr;

                    return ret;
                }
            }

            ythrow yexception() << "shit happen";
        }

        inline void OnError(size_t off, TStringBuf reason) override {
            Y_UNUSED(off);
            if (!FirstErrorReason) {
                FirstErrorReason = reason;
            }
        }

        TConfig Root;
        TConfig* Cur;
        TVector<TConfig> S;
        TString FirstErrorReason;
    };

    TJSONReader r;
    TString data = in.ReadAll();
    TMemoryInput mi(data.data(), data.size());
    TPreprocessor p(g, &mi);

    if (!NJson::ReadJson(&p, false, true, &r)) {
        if (!!r.FirstErrorReason) {
            ythrow TConfigParseError() << "Error parsing json: " << r.FirstErrorReason;
        } else {
            ythrow TConfigParseError() << "Could not parse json " << data.Quote();
        }
    }

    return r.Root;
}

namespace {
    struct TData {
        const char* Prologue;
        const char* Epilogue;
    };

    const TData DATA[] = {
        {"", "\nassert(not (instance == nil))\nreturn instance\n"},
        {"", "\nassert(not (main == nil))\nreturn main\n"},
        {"return ", "\n"},
        {"", "\n"},
    };
}

TConfig TConfig::FromLua(IInputStream& in, const TGlobals& g) {
    const TString& data = in.ReadAll();
    TString json;

    for (size_t i = 0; i < Y_ARRAY_SIZE(DATA); ++i) {
        TStringStream ss;

        ss << TStringBuf("local function configgen()")
           << DATA[i].Prologue << data << DATA[i].Epilogue
           << TStringBuf("end\n\nreturn require('json').encode(configgen())\n");

        try {
            json = TLuaEval().SetVars(g).EvalRaw(ss.Str());

            break;
        } catch (...) {
            if (i == (Y_ARRAY_SIZE(DATA) - 1)) {
                throw;
            }
        }
    }

    TMemoryInput mi(json.data(), json.size());

    return FromJson(mi);
}

TConfig TConfig::FromMarkup(IInputStream& in, const TGlobals& g) {
    TPreprocessor pin(g, &in);

    return ParseRawMarkup(pin);
}

TConfig TConfig::FromIni(IInputStream& in, const TGlobals& g) {
    TPreprocessor pin(g, &in);

    return ParseIni(pin);
}

void TConfig::DumpLua(IOutputStream& out) const {
    TLuaEval e;
    TStringStream ss;

    ToJson(ss);

    e.SetVariable("jsonval", ss.Str());

    out << "return " << e.EvalRaw(TCode::Instance().ObjectByKey("/pp.lua")->ReadAll() + "\nreturn prettify(require('json').decode(jsonval))\n");
}

TConfig TConfig::FromStream(IInputStream& in, const TGlobals& g) {
    const TString& tmp = in.ReadAll();
    TString luaParsingError = "";

    try {
        TMemoryInput mi(tmp.data(), tmp.size());

        return FromLua(mi, g);
    } catch (const yexception& e) {
        luaParsingError = e.AsStrBuf();
    } catch (...) {
        luaParsingError = "unknown error";
    }

    TMemoryInput mi(tmp.data(), tmp.size());

    try {
        return FromJson(mi, g);
    } catch (const yexception& e) {
        const TStringBuf& jsonParsingError = e.AsStrBuf();
        ythrow yexception() << "Could not parse config:\nParsing as lua: " << luaParsingError << "\nParsing as json: " << jsonParsingError;
    }
}

TConfig TConfig::ReadJson(TStringBuf in, const TGlobals& g) {
    TMemoryInput mi(in.data(), in.size());

    return FromJson(mi, g);
}

TConfig TConfig::ReadLua(TStringBuf in, const TGlobals& g) {
    TMemoryInput mi(in.data(), in.size());

    return FromLua(mi, g);
}

TConfig TConfig::ReadMarkup(TStringBuf in, const TGlobals& g) {
    TMemoryInput mi(in.data(), in.size());

    return FromMarkup(mi, g);
}

TConfig TConfig::ReadIni(TStringBuf in, const TGlobals& g) {
    TMemoryInput mi(in.data(), in.size());

    return FromIni(mi, g);
}

void TConfig::Load(IInputStream* input) {
    TString string;
    ::Load(input, string);
    TStringInput stream(string);
    *this = FromJson(stream);
}

void TConfig::Save(IOutputStream* output) const {
    TString result;
    TStringOutput stream(result);
    DumpJson(stream);
    ::Save(output, result);
}

bool TConfig::Has(const TStringBuf& key) const {
    return !operator[](key).IsNull();
}

const TConfig& TConfig::operator[](const TStringBuf& key) const {
    return Get<TDict>().Find(key);
}

const TConfig& TConfig::At(const TStringBuf& key) const {
    return Get<TDict>().At(key);
}

const TConfig& TConfig::operator[](size_t index) const {
    return Get<TArray>().Index(index);
}

size_t TConfig::GetArraySize() const {
    return Get<TArray>().size();
}

const TConfig& TDict::Find(const TStringBuf& key) const {
    const_iterator it = find(key);

    if (it == end()) {
        return Default<TConfig>();
    }

    return it->second;
}

const TConfig& TDict::At(const TStringBuf& key) const {
    const_iterator it = find(key);

    Y_ENSURE_BT(it != end(), "missing key '" << key << "'");

    return it->second;
}

const TConfig& TArray::Index(size_t index) const {
    if (index < size()) {
        return (*this)[index];
    }

    return Default<TConfig>();
}

const TConfig& TArray::At(size_t index) const {
    Y_ENSURE_BT(index < size(), "index " << index << " is out of bounds");

    return (*this)[index];
}

THolder<IInputStream> NConfig::CreatePreprocessor(const TGlobals& g, IInputStream& in) {
    return MakeHolder<TPreprocessor>(g, &in);
}
