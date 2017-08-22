#include <catboost/libs/model/model.h>

int main(int argc, char** argv) {
    if(argc != 3) {
        Cout << "Expected 2 args: ./model_comparator model1 model2" << Endl;
        return -100;
    }
    auto model1 = ReadModel(argv[1]);
    auto model2 = ReadModel(argv[2]);
    if (model1 == model2) {
        Cout << "Models equal" << Endl;
        return 0;
    }
    if (static_cast<TCoreModel&>(model1) != static_cast<TCoreModel&>(model2)) {
        Cout << "Core model differ" << Endl;
        const auto& cm1 = static_cast<TCoreModel&>(model1);
        const auto& cm2 = static_cast<TCoreModel&>(model2);
        if (cm1.Borders != cm2.Borders) {
            Cout << "Borders differ" << Endl;
            auto& b1 = cm1.Borders;
            auto& b2 = cm2.Borders;
            if (b1.size() != b2.size()) {
                Cout << "Sizes differ: " << b1.size() << " != " << b2.size() << Endl;
            } else {
                for (size_t i = 0; i < b1.size(); ++i) {
                    if (b1[i] != b2[i]) {
                        Cout << "Diff for borders in feature " << i << Endl;
                        if (b1[i].size() != b2[i].size()) {
                            Cout << "Sizes differ: " << b1[i].size() << " != " << b2[i].size() << Endl;
                        } else {
                            for (size_t j = 0; j < b1[i].size(); ++j) {
                                if (b1[i][j] != b2[i][j]) {
                                    Cout << j << " " << b1[i][j] << " != " << b2[i][j] << Endl;
                                }
                            }
                        }
                    }
                }
            }
        }
        if (cm1.TreeStruct != cm2.TreeStruct) {
            Cout << "TreeStruct differ" << Endl;
        }
        if (cm1.LeafValues != cm2.LeafValues) {
            Cout << "LeafValues differ" << Endl;
        }
        if (cm1.CatFeatures != cm2.CatFeatures) {
            Cout << "CatFeatures differ" << Endl;
        }
        if (cm1.FeatureIds != cm2.FeatureIds) {
            Cout << "FeatureIds differ" << Endl;
        }
        if (cm1.FeatureCount != cm2.FeatureCount) {
            Cout << "FeatureCount differ" << Endl;
        }
        if (cm1.TargetClassifiers != cm2.TargetClassifiers) {
            Cout << "TargetClassifiers differ" << Endl;
        }
        if (cm1.ModelInfo != cm2.ModelInfo) {
            Cout << "ModelInfo differ" << Endl;
        }
    }
    if (model1.OneHotFeaturesInfo != model2.OneHotFeaturesInfo) {
        Cout << "OneHotFeaturesInfo differ" << Endl;
    }
    if (model1.CtrCalcerData != model2.CtrCalcerData) {
        Cout << "CtrCalcerData differ" << Endl;
    }
    return 1;
}
