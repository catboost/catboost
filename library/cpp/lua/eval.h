#pragma once

#include "wrapper.h"

#include <library/cpp/json/json_value.h>

#include <util/system/mutex.h>

class TLuaEval {
public:
    TLuaEval();

    template <class C>
    inline TLuaEval& SetVars(const C& container) {
        for (auto& [k, v] : container) {
            SetVariable(k, v);
        }

        return *this;
    }

    inline TLuaEval& Parse(TStringBuf chunk) {
        ParseChunk(chunk);

        return *this;
    }

    void SetVariable(TZtStringBuf name, const NJson::TJsonValue& value);
    template <typename T>
    void SetUserdata(TZtStringBuf name, T&& userdata) {
        LuaState_.push_userdata(std::forward<T>(userdata));
        LuaState_.set_global(name.c_str());
    }
    TString EvalExpression(TStringBuf expression);
    TString EvalRaw(TStringBuf code);
    void ParseChunk(TStringBuf code);
    TString Preprocess(TStringBuf line);
    TString PreprocessOne(TStringBuf line);
    bool CheckEmptyStack();

    struct TExpression {
        TString Name;
    };

    TExpression Compile(TStringBuf expression);
    TExpression CompileFunction(TStringBuf expression);
    TExpression CompileRaw(TStringBuf body, const TString& name);
    TString DumpStack();
    TString EvalCompiled(const TExpression& compiled);
    void EvalCompiledRaw(const TExpression& compiled);
    bool EvalCompiledCondition(const TExpression& compiled);
    template <typename TNumber>
    TNumber EvalCompiledNumeric(const TExpression& compiled) {
        TGuard<TMutex> guard(LuaMutex_);
        RunExpressionLocked(guard, compiled);
        return LuaState_.pop_number<TNumber>();
    }

private:
    TString GenerateName();
    TString Evaluate(const TString& name, const TString& body);
    void RunExpressionLocked(const TGuard<TMutex>& lock, const TExpression& compiled);

    TLuaStateHolder LuaState_;
    ui64 FunctionNameCounter_;
    TMutex LuaMutex_;
};
