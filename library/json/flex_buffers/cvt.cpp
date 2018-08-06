#include "cvt.h"

#include <flatbuffers/flexbuffers.h>

#include <library/json/fast_sax/parser.h>
#include <library/json/json_reader.h>

#include <util/generic/vector.h>
#include <util/stream/output.h>
#include <util/stream/input.h>

using namespace NJson;

namespace {
    struct TJsonToFlexCallbacks: public TJsonCallbacks {
        virtual bool OnNull() {
            B.Null();

            return true;
        }

        virtual bool OnBoolean(bool v) {
            B.Bool(v);

            return true;
        }

        virtual bool OnInteger(long long v) {
            B.Int(v);

            return true;
        }

        virtual bool OnUInteger(unsigned long long v) {
            B.UInt(v);

            return true;
        }

        virtual bool OnDouble(double v) {
            B.Double(v);

            return true;
        }

        virtual bool OnString(const TStringBuf& v) {
            B.String(~v, +v);

            return true;
        }

        virtual bool OnOpenMap() {
            S.push_back(B.StartMap());

            return true;
        }

        virtual bool OnMapKey(const TStringBuf& v) {
            B.Key(~v, +v);

            return true;
        }

        virtual bool OnCloseMap() {
            B.EndMap(PopOffset());

            return true;
        }

        virtual bool OnOpenArray() {
            S.push_back(B.StartVector());

            return true;
        }

        virtual bool OnCloseArray() {
            B.EndVector(PopOffset(), false, false);

            return true;
        }

        virtual bool OnStringNoCopy(const TStringBuf& s) {
            return OnString(s);
        }

        virtual bool OnMapKeyNoCopy(const TStringBuf& s) {
            return OnMapKey(s);
        }

        virtual bool OnEnd() {
            B.Finish();

            return true;
        }

        virtual void OnError(size_t, TStringBuf reason) {
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
    };
}

void NJson::ConvertJsonToFlexBuffers(TStringBuf input, TVector<ui8>& result) {
    TJsonToFlexCallbacks cb;

    ReadJsonFast(input, &cb);
    result.swap(const_cast<std::vector<ui8>&>(cb.Buffer()));
}
