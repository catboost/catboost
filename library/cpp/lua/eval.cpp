#include "eval.h"
#include "json.h"
#include <util/string/cast.h>
#include <util/system/guard.h>
#include <util/stream/mem.h>
#include <util/string/builder.h>

TLuaEval::TLuaEval()
    : FunctionNameCounter_(0)
{
    LuaState_.BootStrap();
}

void TLuaEval::SetVariable(TZtStringBuf name, const NJson::TJsonValue& value) {
    TGuard<TMutex> guard(LuaMutex_);

    NLua::PushJsonValue(&LuaState_, value);
    LuaState_.set_global(name.c_str());
}

void TLuaEval::RunExpressionLocked(const TGuard<TMutex>&, const TExpression& expr) {
    LuaState_.push_global(expr.Name.c_str());
    LuaState_.call(0, 1);
}

TString TLuaEval::EvalCompiled(const TExpression& expr) {
    TGuard<TMutex> guard(LuaMutex_);
    RunExpressionLocked(guard, expr);
    return LuaState_.pop_value();
}

void TLuaEval::EvalCompiledRaw(const TExpression& expr) {
    TGuard<TMutex> guard(LuaMutex_);
    RunExpressionLocked(guard, expr);
}

bool TLuaEval::EvalCompiledCondition(const TExpression& expr) {
    TGuard<TMutex> guard(LuaMutex_);
    RunExpressionLocked(guard, expr);
    return LuaState_.pop_bool_strict();
}

TString TLuaEval::EvalRaw(TStringBuf code) {
    TMemoryInput bodyIn(code.data(), code.size());

    LuaState_.Load(&bodyIn, "main");
    LuaState_.call(0, 1);

    return LuaState_.pop_value();
}

void TLuaEval::ParseChunk(TStringBuf code) {
    TMemoryInput in(code.data(), code.size());

    LuaState_.Load(&in, "chunk_" + GenerateName());
    LuaState_.call(0, 0);
}

TString TLuaEval::EvalExpression(TStringBuf expression) {
    const auto expr = Compile(expression);
    try {
        return EvalCompiled(expr);
    } catch (const yexception& e) {
        throw yexception(e) << '\n' << expression;
    }
}

TLuaEval::TExpression TLuaEval::Compile(TStringBuf expression) {
    TGuard<TMutex> guard(LuaMutex_);

    TString name = GenerateName();

    TString body = "function ";
    body += name;
    body += "()\n\treturn (";
    body += expression;
    body += ")\nend\n";

    try {
        TMemoryInput bodyIn(body.c_str(), body.size());
        LuaState_.Load(&bodyIn, "chunk_" + name);
        LuaState_.call(0, 0);
    } catch (const yexception& e) {
        ythrow yexception(e) << "\n"
                             << body;
    }
    return {name};
}

TLuaEval::TExpression TLuaEval::CompileFunction(TStringBuf expression) {
    TString name = GenerateName();
    TStringBuilder body;
    body << "function " << name << "()" << Endl
        << expression << Endl
        << "end";

    return CompileRaw(TStringBuf(body.data(), body.size()), name);
}

TLuaEval::TExpression TLuaEval::CompileRaw(TStringBuf body, const TString& name) {
    TGuard<TMutex> guard(LuaMutex_);
    try {
        TMemoryInput bodyIn(body.data(), body.size());
        LuaState_.Load(&bodyIn, "chunk_" + name);
        LuaState_.call(0, 0);
    } catch (const yexception& e) {
        ythrow yexception(e) << "\n" << body;
    }
    return { name };
}

TString TLuaEval::DumpStack() {
    TString result;
    {
        TStringOutput so(result);
        LuaState_.DumpStack(&so);
    }
    return result;
}

TString TLuaEval::GenerateName() {
    TGuard<TMutex> guard(LuaMutex_);
    return "dummy_" + ToString(FunctionNameCounter_++);
}

template <class T>
static inline T FindEnd(T b, T e) {
    size_t cnt = 0;

    while (b < e) {
        switch (*b) {
            case '{':
                ++cnt;
                break;

            case '}':
                if (cnt == 0) {
                    return b;
                }

                --cnt;
                break;
        }

        ++b;
    }

    return b;
}

TString TLuaEval::PreprocessOne(TStringBuf line) {
    const size_t pos = line.find("${");

    if (pos == TStringBuf::npos) {
        return EvalExpression(line);
    }

    const char* rpos = FindEnd(line.data() + pos + 2, line.end());

    if (rpos == line.end()) {
        ythrow yexception() << TStringBuf("can not parse ") << line;
    }

    const TStringBuf before = line.SubStr(0, pos);
    const TStringBuf after = TStringBuf(rpos + 1, line.end());
    const TStringBuf code = TStringBuf(line.data() + pos + 2, rpos);

    TString res;

    if (code.find("${") == TStringBuf::npos) {
        res = EvalExpression(code);
    } else {
        res = EvalExpression(Preprocess(code));
    }

    return ToString(before) + res + ToString(after);
}

bool TLuaEval::CheckEmptyStack() {
    for (int i = 1; i <= LuaState_.on_stack(); ++i) {
        if (!LuaState_.is_nil(-1 * i)) {
            return false;
        }
    }
    return true;
}

TString TLuaEval::Preprocess(TStringBuf line) {
    TString res = ToString(line);

    while (res.find("${") != TString::npos) {
        res = PreprocessOne(res);
    }

    return res;
}
