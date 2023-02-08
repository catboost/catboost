#pragma once

#include "config.h"

#include <util/string/cast.h>
#include <util/generic/maybe.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>

class IInputStream;

namespace NConfig {
    class IConfig {
    public:
        class IValue {
        public:
            virtual TString AsString() = 0;
            virtual bool AsBool() = 0;
            virtual IConfig* AsSubConfig() = 0;

            virtual bool IsContainer() const = 0;
        };

        class IFunc {
        public:
            inline void Consume(const TString& key, IValue* value) {
                DoConsume(key, value);
            }

            virtual ~IFunc() = default;

        private:
            virtual void DoConsume(const TString& key, IValue* value) {
                (void)key;
                (void)value;
            }
        };

        virtual ~IConfig() = default;

        inline void ForEach(IFunc* func) {
            DoForEach(func);
        }
        virtual void DumpJson(IOutputStream& stream) const = 0;
        virtual void DumpLua(IOutputStream& stream) const = 0;

    private:
        virtual void DoForEach(IFunc* func) = 0;
    };

    template <class T>
    static inline bool ParseFromString(const TString& s, TMaybe<T>& t) {
        t.ConstructInPlace(FromString<T>(s));

        return true;
    }

    template <class T>
    static inline bool ParseFromString(const TString& s, THolder<T>& t) {
        t = MakeHolder<T>(FromString<T>(s));

        return true;
    }

    template <class T>
    static inline bool ParseFromString(const TString& s, T& t) {
        t = FromString<T>(s);

        return true;
    }

    THolder<IConfig> ConfigParser(IInputStream& in, const TGlobals& globals = TGlobals());
}

#define START_PARSE                                                                \
    void DoConsume(const TString& key, NConfig::IConfig::IValue* value) override { \
        (void)key;                                                                 \
        (void)value;
#define END_PARSE                                                                    \
    ythrow NConfig::TConfigParseError() << "unsupported key(" << key.Quote() << ")"; \
    }
#define ON_KEY(k, v) if (key == k && NConfig::ParseFromString(value->AsString(), v))
