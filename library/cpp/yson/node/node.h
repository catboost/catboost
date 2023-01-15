#pragma once

#include <util/generic/bt_exception.h>
#include <util/generic/cast.h>
#include <util/generic/hash.h>
#include <util/generic/variant.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>
#include <util/string/cast.h>

#include <cmath>

class IInputStream;
class IOutputStream;

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

class TNode
{
public:
    class TLookupError
        : public TWithBackTrace<yexception>
    { };

    class TTypeError
        : public TWithBackTrace<yexception>
    { };

    enum EType {
        Undefined = 0   /*"undefined"*/,

        // NOTE: string representation of all node types
        // are compatible with server node type (except `Undefined' which is missing on server).
        String = 1  /*"string_node"*/,
        Int64 = 2   /*"int64_node"*/,
        Uint64 = 3  /*"uint64_node"*/,
        Double = 4  /*"double_node"*/,
        Bool = 5    /*"boolean_node"*/,
        List = 6    /*"list_node"*/,
        Map = 7     /*"map_node"*/,
        Null = 8    /*"null"*/,
    };

    using TListType = TVector<TNode>;
    using TMapType = THashMap<TString, TNode>;

private:
    struct TNull {
        bool operator==(const TNull&) const;
    };

    struct TUndefined {
        bool operator==(const TUndefined&) const;
    };

    using TValue = ::TVariant<
        bool,
        i64,
        ui64,
        double,
        TString,
        TListType,
        TMapType,
        TNull,
        TUndefined
        >;

public:

    TNode();
    TNode(const char* s);
    TNode(const TStringBuf& s);
    TNode(TString s);
    TNode(int i);

    //this case made speccially for prevent mess cast of EType into TNode through TNode(int) constructor
    //usual case of error SomeNode == TNode::Undefined <-- SomeNode indeed will be compared with TNode(0) without this method
    //correct way is SomeNode.GetType() == TNode::Undefined
    template<class T = EType>
    Y_FORCE_INLINE TNode(EType)
    {
        static_assert(!std::is_same<T, EType>::value, "looks like a mistake, may be you forget .GetType()");
    }

    //this case made speccially for prevent mess cast of T* into TNode through implicit bool ctr
    template<class T = int>
    Y_FORCE_INLINE TNode(const T*) : TNode() {
        static_assert(!std::is_same<T,T>::value, "looks like a mistake, and pointer have converted to bool");
    }

    TNode(unsigned int ui);
    TNode(long i);
    TNode(unsigned long ui);
    TNode(long long i);
    TNode(unsigned long long ui);
    TNode(double d);
    TNode(bool b);
    TNode(TMapType map);

    TNode(const TNode& rhs);
    TNode& operator=(const TNode& rhs);

    TNode(TNode&& rhs);
    TNode& operator=(TNode&& rhs);

    ~TNode();

    void Clear();

    bool IsString() const;
    bool IsInt64() const;
    bool IsUint64() const;
    bool IsDouble() const;
    bool IsBool() const;
    bool IsList() const;
    bool IsMap() const;

    // `IsEntity' is deprecated use `IsNull' instead.
    bool IsEntity() const;
    bool IsNull() const;
    bool IsUndefined() const;
    // Returns true if TNode is neither Null, nor Undefined
    bool HasValue() const;

    template<typename T>
    bool IsOfType() const noexcept;

    // Int64, Uint64, Double, or Bool
    bool IsArithmetic() const;

    bool Empty() const;
    size_t Size() const;

    EType GetType() const;

    const TString& AsString() const;
    i64 AsInt64() const;
    ui64 AsUint64() const;
    double AsDouble() const;
    bool AsBool() const;
    const TListType& AsList() const;
    const TMapType& AsMap() const;
    TListType& AsList();
    TMapType& AsMap();

    const TString& UncheckedAsString() const noexcept;
    i64 UncheckedAsInt64() const noexcept;
    ui64 UncheckedAsUint64() const noexcept;
    double UncheckedAsDouble() const noexcept;
    bool UncheckedAsBool() const noexcept;
    const TListType& UncheckedAsList() const noexcept;
    const TMapType& UncheckedAsMap() const noexcept;
    TListType& UncheckedAsList() noexcept;
    TMapType& UncheckedAsMap() noexcept;

    // integer types cast
    // makes overflow checks
    template<typename T>
    T IntCast() const;

    // integers <-> double <-> string
    // makes overflow checks
    template<typename T>
    T ConvertTo() const;

    template<typename T>
    T& As();

    template<typename T>
    const T& As() const;

    static TNode CreateList();
    static TNode CreateList(TListType list);
    static TNode CreateMap();
    static TNode CreateMap(TMapType map);
    static TNode CreateEntity();

    const TNode& operator[](size_t index) const;
    TNode& operator[](size_t index);
    const TNode& At(size_t index) const;
    TNode& At(size_t index);

    TNode& Add() &;
    TNode Add() &&;
    TNode& Add(const TNode& node) &;
    TNode Add(const TNode& node) &&;
    TNode& Add(TNode&& node) &;
    TNode Add(TNode&& node) &&;

    bool HasKey(const TStringBuf key) const;

    TNode& operator()(const TString& key, const TNode& value) &;
    TNode operator()(const TString& key, const TNode& value) &&;
    TNode& operator()(const TString& key, TNode&& value) &;
    TNode operator()(const TString& key, TNode&& value) &&;

    const TNode& operator[](const TStringBuf key) const;
    TNode& operator[](const TStringBuf key);
    const TNode& At(const TStringBuf key) const;
    TNode& At(const TStringBuf key);

    // map getters
    // works the same way like simple getters
    const TString& ChildAsString(const TStringBuf key) const;
    i64 ChildAsInt64(const TStringBuf key) const;
    ui64 ChildAsUint64(const TStringBuf key) const;
    double ChildAsDouble(const TStringBuf key) const;
    bool ChildAsBool(const TStringBuf key) const;
    const TListType& ChildAsList(const TStringBuf key) const;
    const TMapType& ChildAsMap(const TStringBuf key) const;
    TListType& ChildAsList(const TStringBuf key);
    TMapType& ChildAsMap(const TStringBuf key);

    template<typename T>
    T ChildIntCast(const TStringBuf key) const;

    template<typename T>
    T ChildConvertTo(const TStringBuf key) const;

    template<typename T>
    const T& ChildAs(const TStringBuf key) const;

    template<typename T>
    T& ChildAs(const TStringBuf key);

    // list getters
    // works the same way like simple getters
    const TString& ChildAsString(size_t index) const;
    i64 ChildAsInt64(size_t index) const;
    ui64 ChildAsUint64(size_t index) const;
    double ChildAsDouble(size_t index) const;
    bool ChildAsBool(size_t index) const;
    const TListType& ChildAsList(size_t index) const;
    const TMapType& ChildAsMap(size_t index) const;
    TListType& ChildAsList(size_t index);
    TMapType& ChildAsMap(size_t index);

    template<typename T>
    T ChildIntCast(size_t index) const;

    template<typename T>
    T ChildConvertTo(size_t index) const;

    template<typename T>
    const T& ChildAs(size_t index) const;

    template<typename T>
    T& ChildAs(size_t index);


    // attributes
    bool HasAttributes() const;
    void ClearAttributes();
    const TNode& GetAttributes() const;
    TNode& Attributes();

    void MoveWithoutAttributes(TNode&& rhs);

    // Serialize TNode using binary yson format.
    // Methods for ysaveload.
    void Save(IOutputStream* output) const;
    void Load(IInputStream* input);

private:
    void Move(TNode&& rhs);

    void CheckType(EType type) const;

    void AssureMap();
    void AssureList();

    void CreateAttributes();

private:
    TValue Value_;
    THolder<TNode> Attributes_;

    friend bool operator==(const TNode& lhs, const TNode& rhs);
    friend bool operator!=(const TNode& lhs, const TNode& rhs);
};

bool operator==(const TNode& lhs, const TNode& rhs);
bool operator!=(const TNode& lhs, const TNode& rhs);

bool GetBool(const TNode& node);

inline bool TNode::IsArithmetic() const {
    return IsInt64() || IsUint64() || IsDouble() || IsBool();
}

template<typename T>
inline T TNode::IntCast() const {
    if constexpr (std::is_integral<T>::value) {
        try {
            switch (GetType()) {
                case TNode::Uint64:
                    return SafeIntegerCast<T>(AsUint64());
                case TNode::Int64:
                    return SafeIntegerCast<T>(AsInt64());
                default:
                    ythrow TTypeError() << "IntCast() called for type " << GetType();
            }
        } catch(TBadCastException& exc) {
            ythrow TTypeError() << "TBadCastException during IntCast(): " << exc.what();
        }
    } else {
        static_assert(sizeof(T) != sizeof(T), "implemented only for std::is_integral types");
    }
}

template<typename T>
inline T TNode::ConvertTo() const {
    if constexpr (std::is_integral<T>::value) {
        switch (GetType()) {
            case NYT::TNode::String:
                return ::FromString(AsString());
            case NYT::TNode::Int64:
            case NYT::TNode::Uint64:
                return IntCast<T>();
            case NYT::TNode::Double:
                // >= because of (1<<sizeof(T)) + 1
                if (AsDouble() < std::numeric_limits<T>::min() || AsDouble() >= std::numeric_limits<T>::max() || !std::isfinite(AsDouble())) {
                    ythrow TTypeError() << AsDouble() << " can't be converted to " << TypeName<T>();
                }
                return AsDouble();
            case NYT::TNode::Bool:
                return AsBool();
            case NYT::TNode::List:
            case NYT::TNode::Map:
            case NYT::TNode::Null:
            case NYT::TNode::Undefined:
                ythrow TTypeError() << "ConvertTo<" << TypeName<T>() << ">() called for type " << GetType();
        };
    } else {
        static_assert(sizeof(T) != sizeof(T), "should have template specialization");
    }
}

template<>
inline TString TNode::ConvertTo<TString>() const {
    switch (GetType()) {
        case NYT::TNode::String:
            return AsString();
        case NYT::TNode::Int64:
            return ::ToString(AsInt64());
        case NYT::TNode::Uint64:
            return ::ToString(AsUint64());
        case NYT::TNode::Double:
            return ::ToString(AsDouble());
        case NYT::TNode::Bool:
            return ::ToString(AsBool());
        case NYT::TNode::List:
        case NYT::TNode::Map:
        case NYT::TNode::Null:
        case NYT::TNode::Undefined:
            ythrow TTypeError() << "ConvertTo<TString>() called for type " << GetType();
    }
    Y_UNREACHABLE();
}

template<>
inline double TNode::ConvertTo<double>() const {
    switch (GetType()) {
        case NYT::TNode::String:
            return ::FromString(AsString());
        case NYT::TNode::Int64:
            return AsInt64();
        case NYT::TNode::Uint64:
            return AsUint64();
        case NYT::TNode::Double:
            return AsDouble();
        case NYT::TNode::Bool:
            return AsBool();
        case NYT::TNode::List:
        case NYT::TNode::Map:
        case NYT::TNode::Null:
        case NYT::TNode::Undefined:
            ythrow TTypeError() << "ConvertTo<double>() called for type " << GetType();
    }
}

template<>
inline bool TNode::ConvertTo<bool>() const {
    switch (GetType()) {
        case NYT::TNode::String:
            return ::FromString(AsString());
        case NYT::TNode::Int64:
            return AsInt64();
        case NYT::TNode::Uint64:
            return AsUint64();
        case NYT::TNode::Double:
            return AsDouble();
        case NYT::TNode::Bool:
            return AsBool();
        case NYT::TNode::List:
        case NYT::TNode::Map:
        case NYT::TNode::Null:
        case NYT::TNode::Undefined:
            ythrow TTypeError() << "ConvertTo<bool>() called for type " << GetType();
    }
}

template<typename T>
inline T TNode::ChildIntCast(const TStringBuf key) const {
    const auto& node = At(key);
    try {
        return node.IntCast<T>();
    } catch (TTypeError& e) {
        e << ", during getting key=" << key;
        throw e;
    } catch (...) {
        ythrow TTypeError() << CurrentExceptionMessage() << ", during getting key=" << key;
    }
}

template<typename T>
inline T TNode::ChildIntCast(size_t index) const {
    const auto& node = At(index);
    try {
        return node.IntCast<T>();
    } catch (TTypeError& e) {
        e << ", during getting index=" << index;
        throw e;
    } catch (...) {
        ythrow TTypeError() << CurrentExceptionMessage() << ", during getting index=" << index;
    }
}

template<typename T>
inline T TNode::ChildConvertTo(const TStringBuf key) const {
    const auto& node = At(key);
    try {
        return node.ConvertTo<T>();
    } catch (TTypeError& e) {
        e << ", during getting key=" << key;
        throw e;
    } catch (...) {
        ythrow TTypeError() << CurrentExceptionMessage() << ", during getting key=" << key;
    }
}

template<typename T>
inline T TNode::ChildConvertTo(size_t index) const {
    const auto& node = At(index);
    try {
        return node.ConvertTo<T>();
    } catch (TTypeError& e) {
        e << ", during getting index=" << index;
        throw e;
    } catch (...) {
        ythrow TTypeError() << CurrentExceptionMessage() << ", during getting index=" << index;
    }
}

template<typename T>
inline const T& TNode::ChildAs(const TStringBuf key) const {
    const auto& node = At(key);
    try {
        return node.As<T>();
    } catch (TTypeError& e) {
        e << ", during getting key=" << key;
        throw e;
    } catch (...) {
        ythrow TTypeError() << CurrentExceptionMessage() << ", during getting key=" << key;
    }
}

template<typename T>
inline const T& TNode::ChildAs(size_t index) const {
    const auto& node = At(index);
    try {
        return node.As<T>();
    } catch (TTypeError& e) {
        e << ", during getting index=" << index;
        throw e;
    } catch (...) {
        ythrow TTypeError() << CurrentExceptionMessage() << ", during getting index=" << index;
    }
}

template<typename T>
inline T& TNode::ChildAs(const TStringBuf key) {
    return const_cast<T&>(static_cast<const TNode*>(this)->ChildAs<T>(key));
}

template<typename T>
inline T& TNode::ChildAs(size_t index) {
    return const_cast<T&>(static_cast<const TNode*>(this)->ChildAs<T>(index));
}

template<typename T>
inline bool TNode::IsOfType() const noexcept {
    return ::HoldsAlternative<T>(Value_);
}

template<typename T>
inline T& TNode::As() {
    return ::Get<T>(Value_);
}

template<typename T>
inline const T& TNode::As() const {
    return ::Get<T>(Value_);
}

////////////////////////////////////////////////////////////////////////////////

namespace NNodeCmp {
    bool operator<(const TNode& lhs, const TNode& rhs);
    bool operator<=(const TNode& lhs, const TNode& rhs);
    bool operator>(const TNode& lhs, const TNode& rhs);
    bool operator>=(const TNode& lhs, const TNode& rhs);
    bool IsComparableType(const TNode::EType type);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
