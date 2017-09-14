#include <catboost/libs/model/model.h>
#include <catboost/libs/logging/logging.h>

#include <cmath>

template <typename TBorders>
static void BordersDiff(const TBorders& borders1, const TBorders& borders2);

template <typename TLeaves>
static void LeavesDiff(const TLeaves& leaves1, const TLeaves& leaves2);

int main(int argc, char** argv) {
    if(argc != 3) {
        Clog << "Expected 2 args: ./model_comparator model1 model2" << Endl;
        return -100;
    }
    auto model1 = ReadModel(argv[1]);
    auto model2 = ReadModel(argv[2]);
    if (model1 == model2) {
        Clog << "Models equal" << Endl;
        return 0;
    }
    if (static_cast<TCoreModel&>(model1) != static_cast<TCoreModel&>(model2)) {
        Clog << "Core model differ" << Endl;
        const auto& coreModel1 = static_cast<TCoreModel&>(model1);
        const auto& coreModel2 = static_cast<TCoreModel&>(model2);
        if (coreModel1.Borders != coreModel2.Borders) {
            Clog << "Borders differ" << Endl;
            BordersDiff(coreModel1.Borders, coreModel2.Borders);
        }
        if (coreModel1.TreeStruct != coreModel2.TreeStruct) {
            Clog << "TreeStruct differ" << Endl;
        }
        if (coreModel1.LeafValues != coreModel2.LeafValues) {
            LeavesDiff(coreModel1.LeafValues, coreModel2.LeafValues);
        }
        if (coreModel1.CatFeatures != coreModel2.CatFeatures) {
            Clog << "CatFeatures differ" << Endl;
        }
        if (coreModel1.FeatureIds != coreModel2.FeatureIds) {
            Clog << "FeatureIds differ" << Endl;
        }
        if (coreModel1.FeatureCount != coreModel2.FeatureCount) {
            Clog << "FeatureCount differ" << Endl;
        }
        if (coreModel1.TargetClassifiers != coreModel2.TargetClassifiers) {
            Clog << "TargetClassifiers differ" << Endl;
        }
        if (coreModel1.ModelInfo != coreModel2.ModelInfo) {
            Clog << "ModelInfo differ" << Endl;
        }
    }
    if (model1.OneHotFeaturesInfo != model2.OneHotFeaturesInfo) {
        Clog << "OneHotFeaturesInfo differ" << Endl;
    }
    if (model1.CtrCalcerData != model2.CtrCalcerData) {
        Clog << "CtrCalcerData differ" << Endl;
    }
    return 1;
}

template <typename TBorders>
static void BordersDiff(const TBorders& borders1, const TBorders& borders2) {
    if (borders1.size() != borders2.size()) {
        Clog << "Sizes differ: " << borders1.size() << " != " << borders2.size() << Endl;
        return;
    }
    for (size_t borderIdx = 0; borderIdx < borders1.size(); ++borderIdx) {
        if (borders1[borderIdx] != borders2[borderIdx]) {
            Clog << "Diff for borders in feature " << borderIdx << Endl;
            if (borders1[borderIdx].size() != borders2[borderIdx].size()) {
                Clog << "Sizes differ: " << borders1[borderIdx].size() << " != " << borders2[borderIdx].size() << Endl;
            } else {
                for (size_t valueIdx = 0; valueIdx < borders1[borderIdx].size(); ++valueIdx) {
                    if (borders1[borderIdx][valueIdx] != borders2[borderIdx][valueIdx]) {
                        Clog << valueIdx << " " << borders1[borderIdx][valueIdx] << " != " << borders2[borderIdx][valueIdx] << Endl;
                    }
                }
            }
        }
    }
}

template <typename TLeaves>
static void LeavesDiff(const TLeaves& leaves1, const TLeaves& leaves2) {
    double maxEps = 0, maxRelEps = 0;
    for (int treeIdx = 0; treeIdx < Min(leaves1.ysize(), leaves2.ysize()); ++treeIdx) {
        for (int dim = 0; dim < Min(leaves1[treeIdx].ysize(), leaves2[treeIdx].ysize()); ++dim) {
            for (int leafIdx = 0; leafIdx < Min(leaves1[treeIdx][dim].ysize(), leaves2[treeIdx][dim].ysize()); ++leafIdx) {
                const double diff = fabs(leaves1[treeIdx][dim][leafIdx] - leaves2[treeIdx][dim][leafIdx]);
                if (diff > maxEps) {
                    maxEps = diff;
                    const double maxAbsLeaf = Max(fabs(leaves1[treeIdx][dim][leafIdx]), fabs(leaves2[treeIdx][dim][leafIdx]));
                    if (diff / maxAbsLeaf > maxRelEps) {
                        maxRelEps = diff / maxAbsLeaf;
                    }
                }
            } // for leafIdx
            if (leaves1[treeIdx][dim].ysize() != leaves2[treeIdx][dim].ysize()) {
                Clog << "Size differs, " << treeIdx << ", " << dim << Endl;
            }
        } // for dim
        if (leaves1[treeIdx].ysize() != leaves2[treeIdx].ysize()) {
            Clog << "Size differs, " << treeIdx << Endl;
        }
    } // for treeIdx
    if (leaves1.ysize() != leaves2.ysize()) {
        Clog << "Size differs" << Endl;
    }
    Clog << "LeafValues differ: max abs " << maxEps << ", max rel " << maxRelEps << Endl;
}

