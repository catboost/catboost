#include "json_easy_parser.h"
#include <library/cpp/json/json_reader.h>
#include <util/string/cast.h>
#include <util/string/split.h>
#include <util/string/strip.h>

namespace NJson {
    static TString MAP_IDENTIFIER = "{}";
    static TString ARRAY_IDENTIFIER = "[]";
    static TString ANY_IDENTIFIER = "*";

    static void ParsePath(TString path, TVector<TPathElem>* res) {
        TVector<const char*> parts;
        Split(path.begin(), '/', &parts);
        for (size_t n = 0; n < parts.size(); ++n) {
            TString part = Strip(parts[n]);
            if (!part.empty()) {
                if (part[0] != '[') {
                    res->push_back(TPathElem(NImpl::MAP));
                    res->push_back(TPathElem(part));
                } else {
                    int arrayCounter;
                    try {
                        arrayCounter = FromString<int>(part.substr(1, part.length() - 2));
                    } catch (yexception&) {
                        arrayCounter = -1;
                    }
                    res->push_back(TPathElem(arrayCounter));
                }
            }
        }
    }

    void TJsonParser::AddField(const TString& path, bool nonEmpty) {
        Fields.emplace_back();
        Fields.back().NonEmpty = nonEmpty;
        ParsePath(path, &Fields.back().Path);
    }

    TString TJsonParser::ConvertToTabDelimited(const TString& json) const {
        TStringInput in(json);
        TStringStream out;
        ConvertToTabDelimited(in, out);
        return out.Str();
    }

    class TRewriteJsonImpl: public NJson::TJsonCallbacks {
        const TJsonParser& Parent;
        TVector<TString> FieldValues;
        TVector<TPathElem> Stack;
        bool ShouldUpdateOnArrayChange;
        int CurrentFieldIdx;
        bool HasFormatError;

    private:
        static bool PathElementMatch(const TPathElem& templ, const TPathElem& real) {
            if (templ.Type != real.Type)
                return false;
            if (templ.Type == NImpl::ARRAY)
                return templ.ArrayCounter == -1 || templ.ArrayCounter == real.ArrayCounter;
            if (templ.Type == NImpl::MAP_KEY)
                return templ.Key == ANY_IDENTIFIER || templ.Key == real.Key;
            return true;
        }

        bool CheckFilter(const TVector<TPathElem>& path) const {
            if (Stack.size() < path.size())
                return false;
            for (size_t n = 0; n < path.size(); ++n) {
                if (!PathElementMatch(path[n], Stack[n]))
                    return false;
            }
            return true;
        }

        void UpdateRule() {
            for (size_t n = 0; n < Parent.Fields.size(); ++n) {
                if (FieldValues[n].empty() && CheckFilter(Parent.Fields[n].Path)) {
                    CurrentFieldIdx = n;
                    return;
                }
            }
            CurrentFieldIdx = -1;
        }

        void Pop() {
            Stack.pop_back();
        }

        void IncreaseArrayCounter() {
            if (!Stack.empty() && Stack.back().Type == NImpl::ARRAY) {
                ++Stack.back().ArrayCounter;
                if (ShouldUpdateOnArrayChange)
                    UpdateRule();
            }
        }

        template <class T>
        bool OnValue(const T& val) {
            IncreaseArrayCounter();
            if (CurrentFieldIdx >= 0) {
                FieldValues[CurrentFieldIdx] = ToString(val);
                UpdateRule();
            }
            return true;
        }

    public:
        TRewriteJsonImpl(const TJsonParser& parent)
            : Parent(parent)
            , FieldValues(parent.Fields.size())
            , ShouldUpdateOnArrayChange(false)
            , CurrentFieldIdx(-1)
            , HasFormatError(false)
        {
            for (size_t n = 0; n < Parent.Fields.size(); ++n) {
                if (!Parent.Fields[n].Path.empty() && Parent.Fields[n].Path.back().Type == NImpl::ARRAY)
                    ShouldUpdateOnArrayChange = true;
            }
        }

        bool OnOpenMap() override {
            IncreaseArrayCounter();
            Stack.push_back(TPathElem(NImpl::MAP));
            if (CurrentFieldIdx >= 0)
                HasFormatError = true;
            else
                UpdateRule();
            return true;
        }

        bool OnOpenArray() override {
            IncreaseArrayCounter();
            Stack.push_back(TPathElem(-1));
            if (CurrentFieldIdx >= 0)
                HasFormatError = true;
            else
                UpdateRule();
            return true;
        }

        bool OnCloseMap() override {
            while (!Stack.empty() && Stack.back().Type != NImpl::MAP)
                Pop();
            if (!Stack.empty())
                Pop();
            UpdateRule();
            return true;
        }

        bool OnCloseArray() override {
            if (!Stack.empty())
                Pop();
            UpdateRule();
            return true;
        }

        bool OnMapKey(const TStringBuf& key) override {
            if (!Stack.empty() && Stack.back().Type == NImpl::MAP_KEY) {
                Pop();
                UpdateRule();
            }
            Stack.push_back(TPathElem(TString{key}));
            if (CurrentFieldIdx >= 0)
                HasFormatError = true;
            else
                UpdateRule();
            return true;
        }

        bool OnBoolean(bool b) override {
            return OnValue(b);
        }

        bool OnInteger(long long i) override {
            return OnValue(i);
        }

        bool OnDouble(double f) override {
            return OnValue(f);
        }

        bool OnString(const TStringBuf& str) override {
            return OnValue(str);
        }

        bool IsOK() const {
            if (HasFormatError)
                return false;
            for (size_t n = 0; n < FieldValues.size(); ++n)
                if (Parent.Fields[n].NonEmpty && FieldValues[n].empty())
                    return false;
            return true;
        }

        void WriteTo(IOutputStream& out) const {
            for (size_t n = 0; n < FieldValues.size(); ++n)
                out << "\t" << FieldValues[n];
        }

        void WriteTo(TVector<TString>* res) const {
            *res = FieldValues;
        }
    };

    void TJsonParser::ConvertToTabDelimited(IInputStream& in, IOutputStream& out) const {
        TRewriteJsonImpl impl(*this);
        ReadJson(&in, &impl);
        if (impl.IsOK()) {
            out << Prefix;
            impl.WriteTo(out);
            out.Flush();
        }
    }

    bool TJsonParser::Parse(const TString& json, TVector<TString>* res) const {
        TRewriteJsonImpl impl(*this);
        TStringInput in(json);
        ReadJson(&in, &impl);
        if (impl.IsOK()) {
            impl.WriteTo(res);
            return true;
        } else
            return false;
    }

    //struct TTestMe {
    //    TTestMe() {
    //        TJsonParser worker;
    //        worker.AddField("/x/y/z", true);
    //        TString ret1 = worker.ConvertToTabDelimited("{ \"x\" : { \"y\" : { \"w\" : 1, \"z\" : 2 } } }");
    //        TString ret2 = worker.ConvertToTabDelimited(" [1, 2, 3, 4, 5] ");
    //    }
    //} testMe;

}
