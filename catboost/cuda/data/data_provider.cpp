#include "data_provider.h"

namespace NCatboostCuda {

    void TDataProvider::DumpBordersToFileInMatrixnetFormat(const TString& file) {
        TOFStream out(file);
        for (auto& feature : Features) {
            if (feature->GetType() == EFeatureValuesType::BinarizedFloat) {
                    auto binarizedFeaturePtr = dynamic_cast<const TBinarizedFloatValuesHolder*>(feature.Get());
                    auto nanMode = binarizedFeaturePtr->GetNanMode();

                    for (const auto& border : binarizedFeaturePtr->GetBorders()) {
                        out << binarizedFeaturePtr->GetId() << "\t" << ToString<double>(border);
                        if (nanMode != ENanMode::Forbidden) {
                            out << "\t" << nanMode;
                        }
                        out << Endl;
                    }
            }
        }
    }
}
