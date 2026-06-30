#include <library/cpp/getopt/small/last_getopt.h>
#include <library/cpp/json/json_reader.h>

#include <util/generic/set.h>
#include <util/generic/string.h>
#include <util/generic/utility.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/stream/file.h>
#include <util/stream/output.h>
#include <util/string/builder.h>
#include <util/system/compiler.h>

#include <cmath>


static double CalcDiff(double number0, double number1) {
    const double delta = number0 - number1;
    const double maxAbsoluteValue = std::max(std::abs(number0), std::abs(number1));
    const double relativeDiff = maxAbsoluteValue == 0.0 ? 0.0 : std::abs(delta) / maxAbsoluteValue;
    const double minDiff = std::min(std::abs(delta), relativeDiff);
    return minDiff;
}


void OutputWithPathFromRoot(const TVector<TString>& pathFromRoot, const TString& message) {
    Cout << "Diff at";
    for (const auto& pathElement : pathFromRoot) {
        Cout << ' ' << pathElement;
    }
    Cout << " : " << message << Endl;
}


// returns true is values are different
template <class T>
bool CompareJsonValuesSimple(const T& lhs, const T& rhs, const TVector<TString>& pathFromRoot) {
    if (lhs == rhs) {
        return false;
    } else {
        OutputWithPathFromRoot(
            pathFromRoot,
            TStringBuilder() << "Values differ: left: \"" << lhs << "\", right: \"" << rhs << "\"");
        return true;
    }
}

struct TJsonCompareContext {
    double DiffLimit;
    bool StopEarly;
    TVector<TString> PathFromRoot; // TVector, not TStack because iterator is needed
};

bool AreJsonValuesDifferent(
    const NJson::TJsonValue& lhs,
    const NJson::TJsonValue& rhs,
    TJsonCompareContext* context);


bool CompareJsonMaps(
    const NJson::TJsonValue::TMapType& lhs,
    const NJson::TJsonValue::TMapType& rhs,
    TJsonCompareContext* context) {

    TSet<TString> unionKeys;
    for (const auto& [key, value] : lhs) {
        unionKeys.insert(key);
    }
    for (const auto& [key, value] : rhs) {
        unionKeys.insert(key);
    }

    bool diffFound = false;
    for (const auto& key : unionKeys) {
        const auto lhsIt = lhs.find(key);
        if (lhsIt == lhs.end()) {
            OutputWithPathFromRoot(
                context->PathFromRoot,
                TStringBuilder() << "key \"" << key << "\" is present only in the right argument");
            if (context->StopEarly) {
                return true;
            }
            diffFound = true;
        } else {
            const auto rhsIt = rhs.find(key);
            if (rhsIt == rhs.end()) {
                OutputWithPathFromRoot(
                    context->PathFromRoot,
                    TStringBuilder() << "key \"" << key << "\" is present only in the left argument");
                if (context->StopEarly) {
                    return true;
                }
                diffFound = true;
            } else {
                context->PathFromRoot.push_back(TStringBuilder() << "[\"" << key << "\"]");
                if (AreJsonValuesDifferent(lhsIt->second, rhsIt->second, context)) {
                    if (context->StopEarly) {
                        return true;
                    }
                    diffFound = true;
                }
                context->PathFromRoot.pop_back();
            }
        }
    }
    return diffFound;
}


bool CompareJsonArrays(
    const NJson::TJsonValue::TArray& lhs,
    const NJson::TJsonValue::TArray& rhs,
    TJsonCompareContext* context) {

    bool diffFound = false;
    if (lhs.size() != rhs.size()) {
        OutputWithPathFromRoot(
            context->PathFromRoot,
            TStringBuilder() << "Sizes differ: left: " << lhs.size() << ", right: " << rhs.size());
        if (context->StopEarly) {
            return true;
        }
        diffFound = true;
    }

    const auto shorterLength = Min(lhs.size(), rhs.size());

    for (auto i : xrange(shorterLength)) {
        context->PathFromRoot.push_back(TStringBuilder() << '[' << i << ']');
        if (AreJsonValuesDifferent(lhs[i], rhs[i], context)) {
            if (context->StopEarly) {
                return true;
            }
            diffFound = true;
        }
        context->PathFromRoot.pop_back();
    }

    return diffFound;
}


bool AreJsonValuesDifferent(
    const NJson::TJsonValue& lhs,
    const NJson::TJsonValue& rhs,
    TJsonCompareContext* context) {

    if (lhs.GetType() != rhs.GetType()) {
        OutputWithPathFromRoot(
            context->PathFromRoot,
            TStringBuilder() << "Types differ: left: " << lhs.GetType() << ", right: " << rhs.GetType());
        return true;
    } else {
        switch (lhs.GetType()) {
            case NJson::JSON_UNDEFINED:
            case NJson::JSON_NULL:
                return false;
            case NJson::JSON_BOOLEAN:
                return CompareJsonValuesSimple(
                    lhs.GetBooleanSafe(),
                    rhs.GetBooleanSafe(),
                    context->PathFromRoot);
            case NJson::JSON_INTEGER:
                return CompareJsonValuesSimple(
                    lhs.GetIntegerSafe(),
                    rhs.GetIntegerSafe(),
                    context->PathFromRoot);
            case NJson::JSON_DOUBLE:
                {
                    if (CalcDiff(lhs.GetDoubleSafe(), rhs.GetDoubleSafe()) > context->DiffLimit) {
                        OutputWithPathFromRoot(
                            context->PathFromRoot,
                            TStringBuilder() << "Doubles differ greater than limit: left: "
                                << lhs.GetDoubleSafe() << ", right: " << rhs.GetDoubleSafe()
                                << ", diff limit: " << context->DiffLimit);
                        return true;
                    }
                    return false;
                }
            case NJson::JSON_STRING:
                return CompareJsonValuesSimple(lhs.GetStringSafe(), rhs.GetStringSafe(), context->PathFromRoot);
            case NJson::JSON_MAP:
                return CompareJsonMaps(lhs.GetMapSafe(), rhs.GetMapSafe(), context);
            case NJson::JSON_ARRAY:
                return CompareJsonArrays(lhs.GetArraySafe(), rhs.GetArraySafe(), context);
            case NJson::JSON_UINTEGER:
                return CompareJsonValuesSimple(
                    lhs.GetUIntegerSafe(),
                    rhs.GetUIntegerSafe(),
                    context->PathFromRoot);
        }
        Y_UNREACHABLE();
    }
}


static bool AreFilesDifferent(TIFStream& input0, TIFStream& input1, double diffLimit, bool stopEarly) {
    NJson::TJsonValue jsonValue0 = NJson::ReadJsonTree(&input0, /*throwOnError*/ true);
    NJson::TJsonValue jsonValue1 = NJson::ReadJsonTree(&input1, /*throwOnError*/ true);

    TJsonCompareContext context;
    context.DiffLimit = diffLimit;
    context.StopEarly = stopEarly;
    context.PathFromRoot.push_back("root");
    return AreJsonValuesDifferent(jsonValue0, jsonValue1, &context);
}

int main(int argc, const char* argv[]) {
    auto opts = NLastGetopt::TOpts::Default();
    opts.AddHelpOption();
    opts.SetTitle("Compare json files with limited precision comparison for floating-point fields.");

    opts.AddLongOption("diff-limit")
        .RequiredArgument("THRESHOLD")
        .Help("tolerate floating point err less than THRESHOLD (err = min(abs(diff), rel(diff))");

    opts.SetFreeArgsNum(2);
    opts.SetFreeArgTitle(0, "<input-file1>", "Input json file");
    opts.SetFreeArgTitle(1, "<input-file2>", "Input json file");

    NLastGetopt::TOptsParseResult res(&opts, argc, argv);

    const bool stopEarly = res.Has("diff-limit");
    const double diffLimit = res.GetOrElse<double>("diff-limit", 0.0);

    auto inputFileNames = res.GetFreeArgs();

    TIFStream input0(inputFileNames[0]);
    TIFStream input1(inputFileNames[1]);

    return AreFilesDifferent(input0, input1, diffLimit, stopEarly);
}
