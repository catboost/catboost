#include "expression.h"

#include <util/generic/deque.h>
#include <util/generic/strbuf.h>
#include <util/generic/yexception.h>
#include <util/generic/ymath.h>
#include <util/system/unaligned_mem.h>

#include <array>
#include <limits>


static inline bool FastHasPrefix(const TStringBuf& suffix, const TStringBuf& prefix) {
    if (suffix.size() < prefix.size()) {
        return false;
    }

    switch (prefix.size()) {
        case 1:
            return suffix[0] == prefix[0];
        case 2:
            return ReadUnaligned<ui16>(suffix.data()) == ReadUnaligned<ui16>(prefix.data());
    }
    return suffix.StartsWith(prefix);
}

enum TExpressionOperation {
    EO_BEGIN,
    EO_OR = EO_BEGIN, // ||
    EO_AND,           // &&
    EO_LE,            // <=
    EO_L,             // <
    EO_GE,            // >=
    EO_G,             // >
    EO_E,             // ==
    EO_NE,            // !=
    EO_BITS_OR,       // |
    EO_BITS_AND,      // &
    EO_ADD,           // "+"
    EO_SUBSTRACT,     // "-"
    EO_MULTIPLY,      // "*"
    EO_DIVIDE,        // "/"
    EO_END
};

const TStringBuf OperationsStrings[] = {
    TStringBuf("||"),
    TStringBuf("&&"),
    TStringBuf("<="),
    TStringBuf("<"),
    TStringBuf(">="),
    TStringBuf(">"),
    TStringBuf("=="),
    TStringBuf("!="),
    TStringBuf("|"),
    TStringBuf("&"),
    TStringBuf("+"),
    TStringBuf("-"),
    TStringBuf("*"),
    TStringBuf("/")};

const int OperationsPriority[] = {
    0, // "||"
    0, // "&&"
    1, // "<="
    1, // "<"
    1, // ">="
    1, // ">"
    1, // "=="
    1, // "!=
    2, // "|"
    2, // "&"
    3, // "+"
    3, // "-"
    4, // "*"
    4, // "/"
    std::numeric_limits<int>::max()};

static bool FindOperation(const char* expr, size_t a, size_t b, size_t& oa, size_t& ob, TExpressionOperation& oper) {
    size_t balance = 0;
    bool found = false;
    oper = EO_END;
    for (size_t i = b; i > a; --i) {
        if (expr[i - 1] == '"') {
            for (--i; i > a; --i) {
                if (expr[i - 1] == '"') {
                    break;
                }
            }
        } else if (expr[i - 1] == ')') {
            ++balance;
        } else if (expr[i - 1] == '(') {
            if (balance == 0) {
                ythrow yexception() << "CalcExpression error: bad brackets balance";
            }
            --balance;
        } else if (balance == 0) {
            for (TExpressionOperation o = EO_BEGIN; OperationsPriority[o] < OperationsPriority[oper]; o = static_cast<TExpressionOperation>(o + 1)) {
                TStringBuf suffix(&expr[i - 1], b - i + 1);
                if (FastHasPrefix(suffix, OperationsStrings[o])) {
                    ob = i + OperationsStrings[o].size() - 1;
                    oper = o;
                    found = true;
                    oa = i - 1;
                    break;
                }
            }
        }
        if (found && OperationsPriority[oper] == OperationsPriority[EO_BEGIN]) {
            break;
        }
    }
    return found;
}

template <typename T>
T ToNumeric(const TString& str) {
    T value = T();
    TryFromString<T>(str, value);
    return value;
}

static bool IsEmpty(const TString& str) {
    return ToNumeric<double>(str) == 0.0;
}

static bool IsEqual(const TString& a, const TString& b, const double eps) {
    double aVal = 0.0;
    double bVal = 0.0;
    if (TryFromString<double>(a, aVal) && TryFromString<double>(b, bVal)) {
        return fabs(aVal - bVal) < eps;
    }
    // okay, compare as strings
    return a == b;
}

static TString CalcExpression(const char* expr, size_t a, size_t b, const IExpressionAdaptor& data) {
    // Cannot compare values less than 1e-5
    static const double EPS = 1e-5;
    for (; a < b && expr[a] == ' '; ++a) {}
    for (; a < b && expr[b - 1] == ' '; --b) {}
    if (a >= b) {
        return TString();
    }
    size_t oa = 0;
    size_t ob = 0;
    TExpressionOperation op;
    if (FindOperation(expr, a, b, oa, ob, op)) {
        TString vl = CalcExpression(expr, a, oa, data);
        TString vr = CalcExpression(expr, ob, b, data);
        switch (op) {
            case EO_OR:
                return (!IsEmpty(vl) || !IsEmpty(vr)) ? "1.0" : "0.0";
            case EO_AND:
                return (!IsEmpty(vl) && !IsEmpty(vr)) ? "1.0" : "0.0";
            case EO_LE:
                return (ToNumeric<double>(vl) <= ToNumeric<double>(vr) + EPS) ? "1.0" : "0.0";
            case EO_L:
                return (ToNumeric<double>(vl) < ToNumeric<double>(vr) - EPS) ? "1.0" : "0.0";
            case EO_GE:
                return (ToNumeric<double>(vl) >= ToNumeric<double>(vr) - EPS) ? "1.0" : "0.0";
            case EO_G:
                return (ToNumeric<double>(vl) > ToNumeric<double>(vr) + EPS) ? "1.0" : "0.0";
            case EO_E:
                return IsEqual(vl, vr, EPS) ? "1.0" : "0.0";
            case EO_NE:
                return !IsEqual(vl, vr, EPS) ? "1.0" : "0.0";
            case EO_BITS_OR:
                return ToString(static_cast<size_t>(ToNumeric<double>(vl)) | static_cast<size_t>(ToNumeric<double>(vr)));
            case EO_BITS_AND:
                return ToString(static_cast<size_t>(ToNumeric<double>(vl)) & static_cast<size_t>(ToNumeric<double>(vr)));
            case EO_ADD:
                return ToString(ToNumeric<double>(vl) + ToNumeric<double>(vr));
            case EO_SUBSTRACT:
                return ToString(ToNumeric<double>(vl) - ToNumeric<double>(vr));
            case EO_MULTIPLY:
                return ToString(ToNumeric<double>(vl) * ToNumeric<double>(vr));
            case EO_DIVIDE:
                return ToString(ToNumeric<double>(vl) / ToNumeric<double>(vr));

            default:
                ythrow yexception() << "CalcExpression error: can't parse expression";
        }
    } else if (expr[a] == '(') {
        if (expr[b - 1] != ')') {
            ythrow yexception() << "CalcExpression error: extra symbols";
        }
        return CalcExpression(expr, a + 1, b - 1, data);
    } else if (expr[a] == '"') {
        if (expr[b - 1] != '"') {
            ythrow yexception() << "CalcExpression error: extra symbols";
        }
        return TString(&expr[a + 1], &expr[b - 1]);
    } else if (expr[a] == '-') {
        return ToString(-FromString<double>(CalcExpression(expr, a + 1, b, data)));
    } else if (expr[a] == '!') {
        return IsEmpty(CalcExpression(expr, a + 1, b, data)) ? "1.0" : "0.0";
    } else {
        bool isCheckVal = false;
        if (expr[a] == '~') {
            isCheckVal = true;
            ++a;
        }
        TString token(&expr[a], &expr[b]);
        TExpressionVariable val;
        bool found = data.FindValue(token, val);
        if (isCheckVal) {
            return found ? "1.0" : "0.0";
        }
        return found ? val.ToStr() : token;
    }
}

TString CalcExpressionStr(const TString& expr, const IExpressionAdaptor& data) {
    return CalcExpression(expr.c_str(), 0, expr.size(), data);
}

double CalcExpression(const TString& expr, const IExpressionAdaptor& data) {
    return ToNumeric<double>(CalcExpressionStr(expr, data));
}

IExpressionImpl::IExpressionImpl() = default;

IExpressionImpl::~IExpressionImpl() = default;

enum EOperation {
    O_CONST,                // const
    O_TOKEN,                // token
    O_NOT,                  // !
    O_IS_TOKEN,             // ~
    O_MINUS,                // - / unary
    O_EXP,                  // #EXP#
    O_LOG,                  // #LOG#
    O_SQR,                  // #SQR#
    O_SQRT,                 // #SQRT#
    O_SIGMOID,              // #SIGMOID#
    O_HISTOGRAM_PERCENTILE, // #HISTOGRAM_PERCENTILE#
    O_STR_COND,             // ?@ / ternary operator for strings
    O_COND,                 // ?
    O_BINARY_BEGIN,         // binary operator
    O_MIN = O_BINARY_BEGIN, // #MIN#
    O_MAX,                  // #MAX#
    O_OR,                   // ||
    O_AND,                  // &&
    O_STARTS_WITH,          // >? start with
    O_STR_LE,               // <=@ / for string comparsion (alphabetical)
    O_STR_L,                // <@  / for string comparsion (alphabetical)
    O_STR_GE,               // >=@ / for string comparsion (alphabetical)
    O_STR_G,                // >@  / for string comparsion (alphabetical)
    O_VERSION_LE,           // <=# / for version comparsion
    O_VERSION_L,            // <#  / for version comparsion
    O_VERSION_GE,           // >=# / for version comparsion
    O_VERSION_G,            // >#  / for version comparsion
    O_VERSION_E,            // ==# / for version comparsion
    O_VERSION_NE,           // !=# / for version comparsion
    O_LE,                   // <=
    O_L,                    // <
    O_GE,                   // >=
    O_G,                    // >
    O_E,                    // ==
    O_NE,                   // !=
    O_MATCH,                // =~
    O_BITS_OR,              // |
    O_BITS_AND,             // &
    O_ADD,                  // "+"
    O_SUBSTRACT,            // "-"  / binary
    O_MULTIPLY,             // "*"
    O_DIVIDE,               // "/"
    O_POW,                  // "^"
    O_END
};

const TStringBuf EOperationsStrings[] = {
    TStringBuf("const"),
    TStringBuf("token"),
    TStringBuf("!"),
    TStringBuf("~"),
    TStringBuf("-"),
    TStringBuf("#EXP#"),
    TStringBuf("#LOG#"),
    TStringBuf("#SQR#"),
    TStringBuf("#SQRT#"),
    TStringBuf("#SIGMOID#"),
    TStringBuf("#HISTOGRAM_PERCENTILE#"),
    TStringBuf("?@"),
    TStringBuf("?"),
    TStringBuf("#MIN#"),
    TStringBuf("#MAX#"),
    TStringBuf("||"),
    TStringBuf("&&"),
    TStringBuf(">?"),
    TStringBuf("<=@"),
    TStringBuf("<@"),
    TStringBuf(">=@"),
    TStringBuf(">@"),
    TStringBuf("<=#"),
    TStringBuf("<#"),
    TStringBuf(">=#"),
    TStringBuf(">#"),
    TStringBuf("==#"),
    TStringBuf("!=#"),
    TStringBuf("<="),
    TStringBuf("<"),
    TStringBuf(">="),
    TStringBuf(">"),
    TStringBuf("=="),
    TStringBuf("!="),
    TStringBuf("=~"),
    TStringBuf("|"),
    TStringBuf("&"),
    TStringBuf("+"),
    TStringBuf("-"),
    TStringBuf("*"),
    TStringBuf("/"),
    TStringBuf("^")
};

const int EOperationsPriority[] = {
    0, // "const"
    0, // "token"
    0, // "!"
    0, // "~"
    0, // "-"
    0, // "#EXP#"
    0, // "#LOG#"
    0, // "#SQR#"
    0, // "#SQRT#"
    0, // "#SIGMOID#"
    0, // "#HISTOGRAM_PERCENTILE#"
    0, // "?@"
    0, // "?"
    1, // "#MIN#"
    1, // "#MAX#"
    2, // "||"
    2, // "&&"
    3, // ">?"
    3, // ">=@"
    3, // ">@"
    3, // "<=@"
    3, // "<@"
    3, // ">=#"
    3, // ">#"
    3, // "<=#"
    3, // "<#"
    3, // "==#"
    3, // "!=#"
    3, // "<="
    3, // "<"
    3, // ">="
    3, // ">"
    3, // "=="
    3, // "!="
    3, // "=~"
    4, // "|"
    4, // "&"
    5, // "+"
    5, // "-"
    6, // "*"
    6, // "/"
    7, // "^"
    std::numeric_limits<int>::max()
};

constexpr size_t MaxOperands = 3;
const double TExpressionVariable::EPS = 1e-5;

class TExpressionImpl: public IExpressionImpl {
public:
    struct TOperator {
        TOperator(EOperation oper = O_END)
            : Oper(oper)
        {
        }
        TVector<size_t> Input;
        EOperation Oper;
    };

    TString CalcExpressionStr(const IExpressionAdaptor& iadapter) const override;
    double CalcExpression(const IExpressionAdaptor& iadapter) const override;
    TExpressionImpl(TStringBuf expr);
    void GetTokensWithSuffix(const TString& suffix, TVector<TString>& tokens) const override;
    void GetTokensWithPrefix(const TString& prefix, TVector<TString>& tokens) const override;
    void SetRegexMatcher(TExpressionRegexMatcher&& matcher) override {
        Matcher = std::move(matcher);
    }

private:
    TDeque<TOperator> Operations;
    TVector<TString> Consts;
    TVector<TString> Tokens;
    TExpressionRegexMatcher Matcher;

    size_t FindOperation(const TStringBuf& exp, std::array<TStringBuf, MaxOperands>& args, EOperation& oper);
    size_t BuildExpression(TStringBuf str);
    TExpressionVariable CalcVariantExpression(const IExpressionAdaptor& data) const;
};

TExpression::TExpression(TStringBuf expr)
    : Pimpl(new TExpressionImpl(expr))
{
}

static inline void SkipQuoted(const char quote, const TStringBuf& exp, size_t& i) {
    Y_ASSERT(i >= 1);
    if (quote != exp[i - 1]) {
        return;
    }
    bool quoteFound = false;
    Y_ENSURE(i >= 1, "CalcExpression error: Opening quote not found. ");

    i--;
    for (; i > 0; --i) {
        if (quote == exp[i - 1]) {
            quoteFound = true;
            break;
        }
    }
    Y_ENSURE(quoteFound, "CalcExpression error: Opening quote not found. ");
}

void TExpressionImpl::GetTokensWithSuffix(const TString& suffix, TVector<TString>& tokens) const {
    tokens.clear();
    for (const auto& token : Tokens)
        if (token.EndsWith(suffix)) {
            tokens.push_back(token);
        }
}

void TExpressionImpl::GetTokensWithPrefix(const TString& prefix, TVector<TString>& tokens) const {
    tokens.clear();
    for (const auto& token : Tokens) {
        if (token.StartsWith(prefix)) {
            tokens.push_back(token);
        }
    }
}

size_t TExpressionImpl::FindOperation(const TStringBuf& exp, std::array<TStringBuf, MaxOperands>& args, EOperation& oper) {
    size_t parenLevel = 0;
    size_t condLevel = 0;
    size_t condEnd = 0;
    size_t numArgs = 0;
    oper = O_END;
    for (size_t i = exp.size(); i > 0; --i) {
        SkipQuoted('"', exp, i);
        SkipQuoted('\'', exp, i);
        if (exp[i - 1] == ')') {
            ++parenLevel;
        } else if (exp[i - 1] == '(') {
            Y_ENSURE(parenLevel > 0, "TExpression: bad paren balance");
            --parenLevel;
        } else if (parenLevel == 0) {
            TStringBuf suffix = exp.substr(i - 1);

            if (FastHasPrefix(suffix, ":")) {
                ++condLevel;

                if (condLevel == 1) {
                    condEnd = i - 1;
                }
            } else if (FastHasPrefix(suffix, "?@")) {
                Y_ENSURE(condLevel > 0, "TExpression: bad conditional syntax");
                --condLevel;

                if (condLevel == 0) {
                    args[0] = exp.substr(0, i - 1);
                    args[1] = exp.substr(i + 1, condEnd - i - 1);
                    args[2] = exp.substr(condEnd + 1);
                    oper = O_STR_COND;
                    numArgs = 3;
                }
            } else if (FastHasPrefix(suffix, "?")) {
                if (i > 1 && FastHasPrefix(exp.substr(i - 2), ">")) {
                    continue;
                }

                Y_ENSURE(condLevel > 0, "TExpression: bad conditional syntax");
                --condLevel;

                if (condLevel == 0) {
                    args[0] = exp.substr(0, i - 1);
                    args[1] = exp.substr(i, condEnd - i);
                    args[2] = exp.substr(condEnd + 1);
                    oper = O_COND;
                    numArgs = 3;
                }
            } else if (oper != O_COND && oper != O_STR_COND) {
                for (EOperation o = O_BINARY_BEGIN; EOperationsPriority[o] < EOperationsPriority[oper]; o = static_cast<EOperation>(o + 1)) {
                    if (!FastHasPrefix(suffix, EOperationsStrings[o])) {
                        continue;
                    }
                    if (EqualToOneOf(o, O_SUBSTRACT, O_ADD)) { // check if it is unary operations
                        bool isUnary = false;
                        for (size_t j = 1; j < i - 1; ++j) {
                            if (EqualToOneOf(exp[i - 1 - j], ' ', '\n')) {
                                continue;
                            }
                            if (EqualToOneOf(exp[i - 1 - j], '?', ':', '(', '*', '/', '=', '&', '>', '<', '|')) {
                                isUnary = true;
                                break;
                            }
                            isUnary = false;
                            break;
                        }
                        if (isUnary) {
                            continue;
                        }
                    }

                    args[0] = exp.substr(0, i - 1);
                    args[1] = exp.substr(i + EOperationsStrings[o].size() - 1);
                    oper = o;
                    numArgs = 2;
                    break;
                }
            }
        }

        if (numArgs > 0 && oper != O_COND && oper != O_STR_COND && EOperationsPriority[oper] == EOperationsPriority[0]) {
            break;
        }
    }

    return numArgs;
}

static bool Trim(TStringBuf& str) {
    size_t b = 0, e = 0;
    for (e = str.size(); e > 0 && str[e - 1] == ' '; --e) {}
    for (b = 0; b <= e && (b == str.size() || str[b] == ' '); ++b) {}
    str = str.substr(b, e - b);
    return e != b;
}

static size_t InsertOrUpdate(TVector<TString>& values, const TString& v) {
    for (size_t i = 0, size = values.size(); i < size; ++i) {
        if (values[i] == v) {
            return i;
        }
    }
    values.push_back(v);
    return values.size() - 1;
}

size_t TExpressionImpl::BuildExpression(TStringBuf str) {
    Y_ENSURE(Trim(str), "CalcExpression error: empty string. ");
    std::array<TStringBuf, MaxOperands> args;
    EOperation oper;
    size_t order = Operations.size();
    size_t numArgs = FindOperation(str, args, oper);
    if (numArgs) {
        Operations.push_back(TOperator(oper));
        for (size_t i = 0; i < numArgs; ++i) {
            Operations[order].Input.push_back(BuildExpression(args[i]));
        }
        return order;
    } else if (str.size() == 0) {
        Operations.push_back(TOperator(O_TOKEN));
        Operations[order].Input.push_back(InsertOrUpdate(Tokens, ToString(str)));
        return order;
    } else if (str[0] == '(') {
        Y_ENSURE(str.back() == ')', "CalcExpression error: missing closing bracket. ");
        return BuildExpression(str.substr(1, str.size() - 2));
    } else if (str[0] == '"') {
        Y_ENSURE(str.back() == '"' && str.size() >= 2, "CalcExpression error: missing closing quote. ");
        Operations.push_back(TOperator(O_CONST));
        Operations[order].Input.push_back(InsertOrUpdate(Consts, ToString(str.substr(1, str.size() - 2))));
        return order;
    } else if (str[0] == '\'') {
        Y_ENSURE(str.back() == '\'' && str.size() >= 2, "CalcExpression error: missing closing singular quote. ");
        Operations.push_back(TOperator(O_TOKEN));
        Operations[order].Input.push_back(InsertOrUpdate(Tokens, ToString(str.substr(1, str.size() - 2))));
        return order;
    } else if (str[0] == '-') {
        Operations.push_back(TOperator(O_MINUS));
        Operations[order].Input.push_back(BuildExpression(str.substr(1)));
        return order;
    } else if (str[0] == '!') {
        Operations.push_back(TOperator(O_NOT));
        Operations[order].Input.push_back(BuildExpression(str.substr(1)));
        return order;
    } else if (str[0] == '~') {
        Operations.push_back(TOperator(O_IS_TOKEN));
        if (str.back() == '\'' && str[1] == '\'')
            Operations[order].Input.push_back(InsertOrUpdate(Tokens, ToString(str.substr(2, str.size() - 3))));
        else
            Operations[order].Input.push_back(InsertOrUpdate(Tokens, ToString(str.substr(1))));
        return order;
    } else if (str.size() > 4 && str.substr(0, 5) == "#EXP#") {
        Operations.push_back(TOperator(O_EXP));
        Operations[order].Input.push_back(BuildExpression(str.substr(5)));
        return order;
    } else if (str.size() > 4 && str.substr(0, 5) == "#LOG#") {
        Operations.push_back(TOperator(O_LOG));
        Operations[order].Input.push_back(BuildExpression(str.substr(5)));
        return order;
    } else if (str.size() > 4 && str.substr(0, 5) == "#SQR#") {
        Operations.push_back(TOperator(O_SQR));
        Operations[order].Input.push_back(BuildExpression(str.substr(5)));
        return order;
    } else if (str.size() > 5 && str.substr(0, 6) == "#SQRT#") {
        Operations.push_back(TOperator(O_SQRT));
        Operations[order].Input.push_back(BuildExpression(str.substr(6)));
        return order;
    } else if (str.size() > 8 && str.substr(0, 9) == "#SIGMOID#") {
        Operations.push_back(TOperator(O_SIGMOID));
        Operations[order].Input.push_back(BuildExpression(str.substr(9)));
        return order;
    } else if (str.size() > 21 && str.substr(0, 22) == "#HISTOGRAM_PERCENTILE#") {
        Operations.push_back(TOperator(O_HISTOGRAM_PERCENTILE));
        TVector<TStringBuf> splitByComma = StringSplitter(str.substr(22)).Split(',').ToList<TStringBuf>();
        Operations[order].Input.push_back(BuildExpression(splitByComma[0]));
        if (splitByComma.size() > 1) {
            Operations[order].Input.push_back(BuildExpression(str.substr(23 + splitByComma[0].size()))); // take part after comma
        }
        return order;
    } else {
        Operations.push_back(TOperator(O_TOKEN));
        Operations[order].Input.push_back(InsertOrUpdate(Tokens, ToString(str)));
        return order;
    }
}

TExpressionImpl::TExpressionImpl(TStringBuf expr) {
    BuildExpression(expr);
}

TExpressionVariable TExpressionImpl::CalcVariantExpression(const IExpressionAdaptor& data) const {
    TVector<TExpressionVariable> values(Operations.size());
    TExpressionVariable v;
    for (size_t i = Operations.size(); i > 0; --i) {
        switch (Operations[i - 1].Oper) {
            case O_CONST:
                values[i - 1] = Consts[Operations[i - 1].Input.front()];
                break;
            case O_TOKEN:
                if (data.FindValue(Tokens[Operations[i - 1].Input.front()], v))
                    values[i - 1] = v;
                else
                    values[i - 1] = Tokens[Operations[i - 1].Input.front()];
                break;
            case O_IS_TOKEN:
                values[i - 1] = data.FindValue(Tokens[Operations[i - 1].Input.front()], v) ? 1.0 : 0.0;
                break;
            case O_NOT:
                values[i - 1] = values[Operations[i - 1].Input.front()].Not();
                break;
            case O_MINUS:
                values[i - 1] = values[Operations[i - 1].Input.front()].Minus();
                break;
            case O_MIN:
                values[i - 1] = values[Operations[i - 1].Input.front()].Min(values[Operations[i - 1].Input.back()]);
                break;
            case O_MAX:
                values[i - 1] = values[Operations[i - 1].Input.front()].Max(values[Operations[i - 1].Input.back()]);
                break;
            case O_HISTOGRAM_PERCENTILE:
                values[i - 1] = values[Operations[i - 1].Input.front()].HistogramPercentile(values[Operations[i - 1].Input.back()]);
                break;
            case O_OR:
                values[i - 1] = values[Operations[i - 1].Input.front()].Or(values[Operations[i - 1].Input.back()]);
                break;
            case O_AND:
                values[i - 1] = values[Operations[i - 1].Input.front()].And(values[Operations[i - 1].Input.back()]);
                break;
            case O_COND:
                values[i - 1] = values[Operations[i - 1].Input[0]].Cond(
                    values[Operations[i - 1].Input[1]],
                    values[Operations[i - 1].Input[2]]
                );
                break;
            case O_STR_COND:
                values[i - 1] = values[Operations[i - 1].Input[0]].StrCond(
                    values[Operations[i - 1].Input[1]],
                    values[Operations[i - 1].Input[2]]
                );
                break;
            case O_LE:
                values[i - 1] = values[Operations[i - 1].Input.front()].Le(values[Operations[i - 1].Input.back()]);
                break;
            case O_L:
                values[i - 1] = values[Operations[i - 1].Input.front()].L(values[Operations[i - 1].Input.back()]);
                break;
            case O_GE:
                values[i - 1] = values[Operations[i - 1].Input.front()].Ge(values[Operations[i - 1].Input.back()]);
                break;
            case O_G:
                values[i - 1] = values[Operations[i - 1].Input.front()].G(values[Operations[i - 1].Input.back()]);
                break;
            case O_E:
                values[i - 1] = values[Operations[i - 1].Input.front()].E(values[Operations[i - 1].Input.back()]);
                break;
            case O_NE:
                values[i - 1] = values[Operations[i - 1].Input.front()].Ne(values[Operations[i - 1].Input.back()]);
                break;
            case O_MATCH:
                values[i - 1] = Matcher ? TExpressionVariable{Matcher(values[Operations[i - 1].Input.front()].ToStr(), values[Operations[i - 1].Input.back()].ToStr())} : TExpressionVariable{};
                break;
            case O_STARTS_WITH:
                values[i - 1] = values[Operations[i - 1].Input.front()].StrStartsWith(values[Operations[i - 1].Input.back()]);
                break;
            case O_STR_LE:
                values[i - 1] = values[Operations[i - 1].Input.front()].StrLe(values[Operations[i - 1].Input.back()]);
                break;
            case O_STR_L:
                values[i - 1] = values[Operations[i - 1].Input.front()].StrL(values[Operations[i - 1].Input.back()]);
                break;
            case O_STR_GE:
                values[i - 1] = values[Operations[i - 1].Input.front()].StrGe(values[Operations[i - 1].Input.back()]);
                break;
            case O_STR_G:
                values[i - 1] = values[Operations[i - 1].Input.front()].StrG(values[Operations[i - 1].Input.back()]);
                break;
            case O_VERSION_LE:
                values[i - 1] = values[Operations[i - 1].Input.front()].VerLe(values[Operations[i - 1].Input.back()]);
                break;
            case O_VERSION_L:
                values[i - 1] = values[Operations[i - 1].Input.front()].VerL(values[Operations[i - 1].Input.back()]);
                break;
            case O_VERSION_GE:
                values[i - 1] = values[Operations[i - 1].Input.front()].VerGe(values[Operations[i - 1].Input.back()]);
                break;
            case O_VERSION_G:
                values[i - 1] = values[Operations[i - 1].Input.front()].VerG(values[Operations[i - 1].Input.back()]);
                break;
            case O_VERSION_E:
                values[i - 1] = values[Operations[i - 1].Input.front()].VerE(values[Operations[i - 1].Input.back()]);
                break;
            case O_VERSION_NE:
                values[i - 1] = values[Operations[i - 1].Input.front()].VerNe(values[Operations[i - 1].Input.back()]);
                break;
            case O_BITS_OR:
                values[i - 1] = values[Operations[i - 1].Input.front()].BitsOr(values[Operations[i - 1].Input.back()]);
                break;
            case O_BITS_AND:
                values[i - 1] = values[Operations[i - 1].Input.front()].BitsAnd(values[Operations[i - 1].Input.back()]);
                break;
            case O_ADD:
                values[i - 1] = values[Operations[i - 1].Input.front()].Add(values[Operations[i - 1].Input.back()]);
                break;
            case O_SUBSTRACT:
                values[i - 1] = values[Operations[i - 1].Input.front()].Sub(values[Operations[i - 1].Input.back()]);
                break;
            case O_MULTIPLY:
                values[i - 1] = values[Operations[i - 1].Input.front()].Mult(values[Operations[i - 1].Input.back()]);
                break;
            case O_DIVIDE:
                values[i - 1] = values[Operations[i - 1].Input.front()].Div(values[Operations[i - 1].Input.back()]);
                break;
            case O_POW:
                values[i - 1] = values[Operations[i - 1].Input.front()].Pow(values[Operations[i - 1].Input.back()]);
                break;
            case O_EXP:
                values[i - 1] = values[Operations[i - 1].Input.front()].Exp();
                break;
            case O_LOG:
                values[i - 1] = values[Operations[i - 1].Input.front()].Log();
                break;
            case O_SQR:
                values[i - 1] = values[Operations[i - 1].Input.front()].Sqr();
                break;
            case O_SQRT:
                values[i - 1] = values[Operations[i - 1].Input.front()].Sqrt();
                break;
            case O_SIGMOID:
                values[i - 1] = values[Operations[i - 1].Input.front()].Sigmoid();
                break;

            default:
                ythrow yexception() << "CalcExpression error: can't parse expression";
        }
    }
    return values.front();
}

double TExpressionImpl::CalcExpression(const IExpressionAdaptor& data) const {
    return CalcVariantExpression(data).ToDouble();
}

TString TExpressionImpl::CalcExpressionStr(const IExpressionAdaptor& data) const {
    return CalcVariantExpression(data).ToStr();
}

void TExpression::GetTokensWithSuffix(const TString& suffix, TVector<TString>& tokens) const {
    Pimpl->GetTokensWithSuffix(suffix, tokens);
}

void TExpression::GetTokensWithPrefix(const TString& prefix, TVector<TString>& tokens) const {
    Pimpl->GetTokensWithPrefix(prefix, tokens);
}

TExpression& TExpression::SetRegexMatcher(TExpressionRegexMatcher matcher) {
    Pimpl->SetRegexMatcher(std::move(matcher));
    return *this;
}
