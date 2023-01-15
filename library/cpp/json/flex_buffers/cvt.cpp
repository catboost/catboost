#include "cvt.h"

#include <flatbuffers/flexbuffers.h>

#include <library/cpp/json/fast_sax/parser.h>
#include <library/cpp/json/json_reader.h>

#include <util/generic/vector.h>
#include <util/stream/output.h>
#include <util/stream/input.h>
#include <util/memory/pool.h>

using namespace NJson;

namespace {
    struct TJsonToFlexCallbacks: public TJsonCallbacks {
        inline TJsonToFlexCallbacks()
            : P(8192)
        {
        }

        bool OnNull() override {
            B.Null();

            return true;
        }

        bool OnBoolean(bool v) override {
            B.Bool(v);

            return true;
        }

        bool OnInteger(long long v) override {
            B.Int(v);

            return true;
        }

        bool OnUInteger(unsigned long long v) override {
            B.UInt(v);

            return true;
        }

        bool OnDouble(double v) override {
            B.Double(v);

            return true;
        }

        bool OnString(const TStringBuf& v) override {
            B.String(v.data(), v.size());

            return true;
        }

        bool OnOpenMap() override {
            S.push_back(B.StartMap());

            return true;
        }

        bool OnMapKey(const TStringBuf& v) override {
            auto iv = P.AppendCString(v);

            B.Key(iv.data(), iv.size());

            return true;
        }

        bool OnCloseMap() override {
            B.EndMap(PopOffset());

            return true;
        }

        bool OnOpenArray() override {
            S.push_back(B.StartVector());

            return true;
        }

        bool OnCloseArray() override {
            B.EndVector(PopOffset(), false, false);

            return true;
        }

        bool OnStringNoCopy(const TStringBuf& s) override {
            return OnString(s);
        }

        bool OnMapKeyNoCopy(const TStringBuf& s) override {
            return OnMapKey(s);
        }

        bool OnEnd() override {
            B.Finish();

            Y_ENSURE(S.empty());

            return true;
        }

        void OnError(size_t, TStringBuf reason) override {
            ythrow yexception() << reason;
        }

        inline size_t PopOffset() {
            auto res = S.back();

            S.pop_back();

            return res;
        }

        inline auto& Buffer() {
            return B.GetBuffer();
        }

        flexbuffers::Builder B;
        TVector<size_t> S;
        TMemoryPool P;
    };
}

void NJson::ConvertJsonToFlexBuffers(TStringBuf input, TFlexBuffersData& result) {
    TJsonToFlexCallbacks cb;

    ReadJsonFast(input, &cb);
    result.swap(const_cast<std::vector<ui8>&>(cb.Buffer()));
}

TString NJson::FlexToString(const TFlexBuffersData& v) {
    auto root = flexbuffers::GetRoot(v.data(), v.size());

    return TString(root.ToString());
}
