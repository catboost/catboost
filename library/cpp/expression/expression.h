#pragma once

#include <util/generic/maybe.h>
#include <util/generic/map.h>
#include <util/generic/set.h>
#include <util/generic/hash.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/string/builder.h>

#include "histogram_points_and_bins.h"
#include "expression_variable.h"


class IExpressionAdaptor {
private:
    IExpressionAdaptor(const IExpressionAdaptor&);
    IExpressionAdaptor& operator=(const IExpressionAdaptor&);

protected:
    virtual ~IExpressionAdaptor() = default;
    IExpressionAdaptor() = default;

public:
    virtual bool FindValue(const TString& name, TExpressionVariable& value) const = 0;
};

template <typename T>
class TExpressionAdaptor: public IExpressionAdaptor {
private:
    const T& Container;

public:
    TExpressionAdaptor(const T& container)
        : Container(container)
    {
    }
    bool FindValue(const TString& name, TExpressionVariable& value) const override {
        typename T::const_iterator it = Container.find(name);
        if (it == Container.end()) {
            return false;
        }
        value = it->second;
        return true;
    }
};

template <typename... Ts>
class TExpressionMultiAdaptorImpl {
public:
    bool FindValue(const TString& /*name*/, TExpressionVariable& /*value*/) const {
        return false;
    }

    template <typename T>
    TMaybe<T> FindOnce(const TString& /*name*/) const {
        return Nothing();
    }

    template <typename T>
    TMaybe<T> FindSum(const TString& /*name*/) const {
        return Nothing();
    }
};

template <typename T, typename... Ts>
class TExpressionMultiAdaptorImpl<T, Ts...> : public TExpressionMultiAdaptorImpl<Ts...>
{
private:
    const T& Container;
    using TValueType = typename T::mapped_type;

public:
    TExpressionMultiAdaptorImpl(const T& container, const Ts&... otherContainers)
        : TExpressionMultiAdaptorImpl<Ts...>(otherContainers...)
        , Container(container)
    {
    }

    template <typename TReturnType>
    TMaybe<TReturnType> FindOnce(const TString& name) const {
        typename T::const_iterator it = Container.find(name);
        if (it == Container.end()) {
            return TExpressionMultiAdaptorImpl<Ts...>::template FindOnce<TReturnType>(name);
        }

        return it->second;
    }

    template <typename TReturnType>
    TMaybe<TReturnType> FindSum(const TString& name) const {
        const auto value = TExpressionMultiAdaptorImpl<Ts...>::template FindSum<TReturnType>(name);

        typename T::const_iterator it = Container.find(name);
        if (it != Container.end()) {
            if (value.Defined()) {
                return *value + it->second;
            } else {
                return it->second;
            }
        }
        return value;
    }
};

template <typename... Ts>
class TExpressionMultiAdaptor : public IExpressionAdaptor {
protected:
    TExpressionMultiAdaptorImpl<Ts...> Impl;

public:
    TExpressionMultiAdaptor(const Ts&... ts)
        : Impl(ts...)
    {}

    template <typename T>
    TMaybe<T> FindOnce(const TString& name) const {
        return Impl.template FindOnce<T>(name);
    }

    bool FindValue(const TString& name, TExpressionVariable& valueOut) const override {
        const auto value = FindOnce<double>(name);
        const bool found = value.Defined();

        if (found) {
            valueOut = *value;
        }

        return found;
    }

    const IExpressionAdaptor& AsInterface() const {
        return *this;
    }
};

template <typename... Ts>
class TExpressionSummedMultiAdaptor : public IExpressionAdaptor {
protected:
    TExpressionMultiAdaptorImpl<Ts...> Impl;

public:
    TExpressionSummedMultiAdaptor(const Ts&... ts)
        : Impl(ts...)
    {}

    template <typename T>
    TMaybe<T> FindSum(const TString& name) const {
        return Impl.template FindSum<T>(name);
    }

    bool FindValue(const TString& name, TExpressionVariable& valueOut) const override {
        const auto value = FindSum<double>(name);
        const bool found = value.Defined();

        if (found) {
            valueOut = *value;
        }

        return found;
    }

    const IExpressionAdaptor& AsInterface() const {
        return *this;
    }
};

TString CalcExpressionStr(const TString& expr, const IExpressionAdaptor& data);
double CalcExpression(const TString& expr, const IExpressionAdaptor& data);

using TExpressionRegexMatcher = std::function<bool(TStringBuf str, TStringBuf rx)>;

class IExpressionImpl {
public:
    IExpressionImpl();
    virtual ~IExpressionImpl();
    virtual TString CalcExpressionStr(const IExpressionAdaptor& iadapter) const = 0;
    virtual double CalcExpression(const IExpressionAdaptor& iadapter) const = 0;
    virtual void GetTokensWithSuffix(const TString& suffix, TVector<TString>& tokens) const = 0;
    virtual void GetTokensWithPrefix(const TString& prefix, TVector<TString>& tokens) const = 0;
    virtual void SetRegexMatcher(TExpressionRegexMatcher&& matcher) = 0;
};

class TExpression {
private:
    TSimpleSharedPtr<IExpressionImpl> Pimpl;

public:
    explicit TExpression(TStringBuf expr);
    template <typename T>
    double CalcExpression(const T& container) const {
        const TExpressionAdaptor<T> adapter(container);
        const IExpressionAdaptor& iadapter(adapter);
        return Pimpl->CalcExpression(iadapter);
    }

    template <typename T>
    TString CalcExpressionStr(const T& container) const {
        const TExpressionAdaptor<T> adapter(container);
        const IExpressionAdaptor& iadapter(adapter);
        return Pimpl->CalcExpressionStr(iadapter);
    }
    double CalcAdapterExpression(const IExpressionAdaptor& iadapter) const {
        return Pimpl->CalcExpression(iadapter);
    }
    void GetTokensWithSuffix(const TString& suffix, TVector<TString>& tokens) const;
    void GetTokensWithPrefix(const TString& prefix, TVector<TString>& tokens) const;
    TExpression& SetRegexMatcher(TExpressionRegexMatcher matcher);
};

template<>
inline double TExpression::CalcExpression<IExpressionAdaptor>(const IExpressionAdaptor& iadapter) const {
    return Pimpl->CalcExpression(iadapter);
}

template<>
inline TString TExpression::CalcExpressionStr<IExpressionAdaptor>(const IExpressionAdaptor& iadapter) const {
    return Pimpl->CalcExpressionStr(iadapter);
}

template <typename T>
double CalcExpression(const TString& expr, const T& container) {
    TExpression exp(expr);
    return exp.CalcExpression(container);
}

template <typename T>
TString CalcExpressionStr(const TString& expr, const T& container) {
    TExpression exp(expr);
    return exp.CalcExpressionStr(container);
}

class TOptionalExpression {
public:
    TOptionalExpression(TMaybe<TString> expressionStr)
        : ExpressionStr(expressionStr) {}

    void Init() {
        if (ExpressionStr.Defined()) {
            Expression.Reset(new TExpression(ExpressionStr.GetRef()));
        }
    }

    bool CheckAdapterCondition(const IExpressionAdaptor& iadapter, bool defaultValue = true) const {
        if (Expression.Get() == nullptr) {
            return defaultValue;
        }

        return 0.0 != Expression->CalcAdapterExpression(iadapter);
    }

    double CalcAdapterExpression(const IExpressionAdaptor& iadapter, double defautlValue = 1.0) const {
        if (Expression.Get() == nullptr) {
            return defautlValue;
        }

        return Expression->CalcAdapterExpression(iadapter);
    }


    bool HasExpression() const {
        return nullptr != GetExpression();
    }

    const TMaybe<TString> GetExpressionStr() const {
        return ExpressionStr;
    }

    const TExpression* GetExpression() const {
        return Expression.Get();
    }

private:
    TMaybe<TString> ExpressionStr;
    THolder<TExpression> Expression;
};
