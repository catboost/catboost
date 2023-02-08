#include "markup.h"

#include <util/stream/output.h>
#include <util/stream/mem.h>
#include <util/string/strip.h>
#include <util/string/cast.h>

using namespace NConfig;

#define DBG(x)

namespace {
#define MACHINE_DATA
#include "markupfsm.h"
#undef MACHINE_DATA

    class IXmlCB {
    public:
        inline void DoTagOpen(const TStringBuf& key) {
            DBG(Cerr << "topen" << key << Endl);

            S_.push_back(key);
            OnTagOpen(key);
        }

        inline void DoTagClose(const TStringBuf& key) {
            DBG(Cerr << "tclose" << key << Endl);

            if (S_.empty()) {
                ythrow yexception() << "unbalanced tag";
            }

            if (S_.back() != key) {
                ythrow yexception() << "unbalanced tag";
            }

            S_.pop_back();
            OnTagClose();
        }

        inline void DoText(const TStringBuf& text) {
            DBG(Cerr << "ttext" << text << Endl);

            if (!!text) {
                OnText(text);
            }
        }

        inline void DoAttrKey(const TStringBuf& key) {
            DBG(Cerr << "tattrkey" << key << Endl);

            A_ = key;
        }

        inline void DoAttrValue(const TStringBuf& key) {
            DBG(Cerr << "tattrvalue" << key << Endl);

            if (!A_) {
                ythrow yexception() << "dangling attribute";
            }

            OnAttr(A_, key);
            A_ = TStringBuf();
        }

        virtual void OnTagOpen(const TStringBuf& key) = 0;
        virtual void OnTagClose() = 0;
        virtual void OnText(const TStringBuf& text) = 0;
        virtual void OnAttr(const TStringBuf& key, const TStringBuf& value) = 0;
        virtual ~IXmlCB() = default;

    private:
        TVector<TStringBuf> S_;
        TStringBuf A_;
    };

    inline void Parse(TStringBuf s, IXmlCB* cb) {
        const char* p = s.data();
        const char* pe = s.end();
        const char* eof = pe;
        const char* l = p;

        int cs;
        TString cur;

#define MACHINE_INIT
#include "markupfsm.h"
#undef MACHINE_INIT

#define MACHINE_EXEC
#include "markupfsm.h"
#undef MACHINE_EXEC

        if (cs < ParseXml_first_final) {
            ythrow TConfigParseError() << "can not parse markup data at offset " << (p - s.data());
        }
    }

    inline IValue* SmartValue(const TStringBuf& v) {
        try {
            return ConstructValue(FromString<ui64>(v));
        } catch (...) {
        }

        try {
            return ConstructValue(FromString<i64>(v));
        } catch (...) {
        }

        try {
            return ConstructValue(FromString<double>(v));
        } catch (...) {
        }

        try {
            return ConstructValue(FromString<bool>(v));
        } catch (...) {
        }

        return ConstructValue(ToString(v));
    }

    inline TConfig Parse(TStringBuf s0) {
        struct TXmlParser: public IXmlCB {
            inline TXmlParser()
                : Root(ConstructValue(TDict()))
            {
                S.push_back(&Root);
            }

            void OnTagOpen(const TStringBuf& key) override {
                *Push(key) = ConstructValue(TDict());
            }

            void OnTagClose() override {
                S.pop_back();
            }

            static inline bool IsWS(char ch) {
                switch (ch) {
                    case ' ':
                    case '\t':
                    case '\r':
                    case '\n':
                    case ':':
                        return true;
                }

                return false;
            }

            void OnText(const TStringBuf& text) override {
                TMemoryInput mi(text.data(), text.size());
                TString line;

                while (mi.ReadLine(line)) {
                    DBG(Cerr << line << Endl);

                    TStringBuf s = StripString(TStringBuf(line));

                    DBG(Cerr << s << Endl);

                    if (!s) {
                        continue;
                    }

                    const char* b = s.data();
                    const char* c = b;
                    const char* e = s.end();

                    while (c < e && !IsWS(*c)) {
                        ++c;
                    }

                    const TStringBuf key(b, c);

                    while (c < e && IsWS(*c)) {
                        ++c;
                    }

                    const TStringBuf value(c, e);

                    if (!key) {
                        continue;
                    }

                    DBG(Cerr << key << " " << value << Endl);

                    SetAttr(key, value);
                }
            }

            void OnAttr(const TStringBuf& key, const TStringBuf& value) override {
                SetAttr(key, value);
            }

            inline void SetAttr(const TStringBuf& key, const TStringBuf& value) {
                Dict()[ToString(key)] = SmartValue(value);
            }

            inline TConfig* Top() {
                return S.back();
            }

            inline TConfig* Push(const TStringBuf& key) {
                TDict& d = Dict();
                const TString k = ToString(key);

                if (d.find(k) == d.end()) {
                    S.push_back(&d[k]);
                } else {
                    TConfig tmp = d[k];

                    if (tmp.IsA<TArray>()) {
                        TArray& arr = d[k].GetNonConstant<TArray>();

                        arr.push_back(TConfig());

                        S.push_back(&arr.back());
                    } else {
                        d[k] = ConstructValue(TArray());

                        TArray& arr = d[k].GetNonConstant<TArray>();

                        arr.push_back(tmp);
                        arr.push_back(TConfig());

                        S.push_back(&arr.back());
                    }
                }

                return Top();
            }

            inline TDict& Dict() {
                try {
                    return Top()->GetNonConstant<TDict>();
                } catch (...) {
                }

                return Top()->Get<TArray>().back().GetNonConstant<TDict>();
            }

            TConfig Root;
            TVector<TConfig*> S;
        };

        TXmlParser cb;

        Parse(s0, &cb);

        return cb.Root;
    }
}

TConfig NConfig::ParseRawMarkup(IInputStream& in) {
    return Parse(in.ReadAll());
}
