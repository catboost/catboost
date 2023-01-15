%module native_impl

%{
#include <catboost/spark/catboost4j-spark/src/native_impl/quantization.h>
#include <util/generic/yexception.h>
%}

%include "enums.swg"
%javaconst(1);
enum class EBorderSelectionType {
    Median = 1,
    GreedyLogSum = 2,
    UniformAndQuantiles = 3,
    MinEntropy = 4,
    MaxLogSum = 5,
    Uniform = 6,
    GreedyMinEntropy = 7,
};

template <class T>
struct TDefaultValue {
    T Value;
    ui64 Count;

public:
    TDefaultValue(T value, ui64 count)
        : Value(value)
        , Count(count)
    {
        Y_ENSURE(Count >= 1, "It is required that default value count is non-0");
    }
};

%template(TDefaultValue_float) TDefaultValue<float>;

%include "cpointer.i"
%pointer_class(float, floatp)
%pointer_class(bool, boolp)

namespace NSplitSelection {
    struct TDefaultQuantizedBin {
        ui32 Idx; // if for splits: bin borders are [Border[Idx - 1], Borders[Idx])
        float Fraction;
    
    public:
        bool operator==(const TDefaultQuantizedBin & rhs) const {
            constexpr float EPS = 1e-6f;
            return (Idx == rhs.Idx) && (Abs(Fraction - rhs.Fraction) < EPS);
        }
    };
}

%include "tvector.i"

%template(FloatVector) TVector<float>;

%include <typemaps.i>

%typemap(throws, throws="Exception") yexception {
    jclass excep = jenv->FindClass("ai/catboost/CatBoostError");
    if (excep)
        jenv->ThrowNew(excep, $1.what());
    return $null;
}

%include <catboost/spark/catboost4j-spark/src/native_impl/quantization.h>
