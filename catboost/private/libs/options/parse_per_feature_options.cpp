#include "parse_per_feature_options.h"

namespace NCatboostOptions {
    std::regex GetDenseFormatPattern(const TStringBuf featureOptionRegex) { // like: (1,0,0,-1,0)
        TString patternStr = "^\\((";
        patternStr += featureOptionRegex;
        patternStr += ")(,(";
        patternStr += featureOptionRegex;
        patternStr += "))*\\)$";
        return std::regex(patternStr.data());
    }

    std::regex GetSparseFormatPattern(const TStringBuf featureOptionRegex) { // like: 0:1,3:-1 or FeatureName1:-1,FeatureName2:-1
        TString patternStr = "^[^:,]+:(";
        patternStr += featureOptionRegex;
        patternStr += ")(,[^:,]+:(";
        patternStr += featureOptionRegex;
        patternStr += "))*$";
        return std::regex(patternStr.data());
    }
}