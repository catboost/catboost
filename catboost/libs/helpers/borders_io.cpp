#include "borders_io.h"
#include "exception.h"

#include <util/string/cast.h>
#include <util/string/split.h>


namespace NCB {
    void ParseBordersFileLine(
        const TString& line,
        ui32* flatFeatureIdx,
        float* border,
        TMaybe<ENanMode>* nanMode
    ) {
        TVector<TString> tokens;
        StringSplitter(line).Split('\t').SkipEmpty().Collect(&tokens);
        CB_ENSURE(
            tokens.ysize() == 2 || tokens.ysize() == 3,
            "Each line should have two or three columns");
        *flatFeatureIdx = FromString<ui32>(tokens[0]);

        *border = FromString<float>(tokens[1]);

        *nanMode = Nothing();
        if (tokens.ysize() == 3) {
            *nanMode = FromString<ENanMode>(tokens[2]);
        }
    }

    void OutputFeatureBorders(
        ui32 flatFeatureIdx,
        const TVector<float>& borders,
        ENanMode nanMode,
        IOutputStream* output
    ) {
        for (const auto& border : borders) {
            (*output) << flatFeatureIdx << "\t" << ToString<double>(border);
            if (nanMode != ENanMode::Forbidden) {
                (*output) << "\t" << nanMode;
            }
            (*output) << Endl;
        }
    }
}

