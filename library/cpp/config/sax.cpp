#include "sax.h"

using namespace NConfig;

namespace {
    class TSax: public IConfig, public IConfig::IValue {
    public:
        inline TSax(const TConfig& cfg)
            : C_(cfg)
        {
        }

        void DoForEach(IFunc* func) override {
            if (C_.IsA<TArray>()) {
                const TArray& a = C_.Get<TArray>();

                for (size_t i = 0; i < a.size(); ++i) {
                    TSax slave(a[i]);

                    func->Consume(ToString(i), &slave);
                }
            } else if (C_.IsA<TDict>()) {
                const TDict& d = C_.Get<TDict>();

                for (const auto& it : d) {
                    TSax slave(it.second);

                    func->Consume(it.first, &slave);
                }
            }
        }

        TString AsString() override {
            if (C_.IsA<TArray>()) {
                TSax slave(C_.Get<TArray>()[0]);
                return slave.AsString();
            }
            return C_.As<TString>();
        }

        bool AsBool() override {
            return C_.As<bool>();
        }

        IConfig* AsSubConfig() override {
            return this;
        }
        bool IsContainer() const override {
            return C_.IsA<TArray>() || C_.IsA<TDict>();
        }
        void DumpJson(IOutputStream& stream) const override {
            C_.DumpJson(stream);
        }
        void DumpLua(IOutputStream& stream) const override {
            C_.DumpLua(stream);
        }

    private:
        TConfig C_;
    };
}

namespace NConfig {
    THolder<IConfig> ConfigParser(IInputStream& in, const TGlobals& globals) {
        return MakeHolder<TSax>(TConfig::FromStream(in, globals));
    }
}
