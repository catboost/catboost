#include <catboost/libs/model/model.h>
#include <catboost/libs/logging/logging.h>

#include <cmath>

template <typename TBorders>
static void FeatureBordersDiff(size_t featureId, const TBorders& borders1, const TBorders& borders2) {
    if (borders1.size() != borders2.size()) {
        Clog << "Feature " << featureId << " borders sizes differ: " << borders1.size() << " != " << borders2.size() << Endl;
        return;
    }
    Clog << "Diff for borders in feature " << featureId << Endl;

    for (size_t valueIdx = 0; valueIdx < borders1.size(); ++valueIdx) {
        if (borders1[valueIdx] != borders2[valueIdx]) {
            Clog << valueIdx << " " << borders1[valueIdx] << " != " << borders2[valueIdx] << Endl;
        }
    }
}

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
        Clog << "Models are equal" << Endl;
        return 0;
    }
    if (model1.ObliviousTrees.FloatFeatures != model2.ObliviousTrees.FloatFeatures) {
        Clog << "FloatFeatures differ" << Endl;
        if (model1.ObliviousTrees.FloatFeatures.size() != model2.ObliviousTrees.FloatFeatures.size()) {
            Clog << "FloatFeatures size differ" << Endl;
        } else {
            for (size_t i = 0; i < model1.ObliviousTrees.FloatFeatures.size(); ++i) {
                auto& floatFeature1 = model1.ObliviousTrees.FloatFeatures[i];
                auto& floatFeature2 = model2.ObliviousTrees.FloatFeatures[i];
                FeatureBordersDiff(i, floatFeature1.Borders, floatFeature2.Borders);
                if (floatFeature1.FeatureId != floatFeature2.FeatureId) {
                    Clog << "FloatFeature FeatureId differ" << Endl;
                }
            }
        }
    }
     if (model1.ObliviousTrees.CatFeatures != model2.ObliviousTrees.CatFeatures) {
        Clog << "CatFeatures differ" << Endl;
     }
     if (model1.ObliviousTrees.OneHotFeatures != model2.ObliviousTrees.OneHotFeatures) {
        Clog << "OneHotFeatures differ" << Endl;
        if (model1.ObliviousTrees.OneHotFeatures.size() != model2.ObliviousTrees.OneHotFeatures.size()) {
            Clog << "OneHotFeatures size differ" << Endl;
        } else {
            for (size_t i = 0; i < model1.ObliviousTrees.OneHotFeatures.size(); ++i) {
                auto& feature1 = model1.ObliviousTrees.OneHotFeatures[i];
                auto& feature2 = model2.ObliviousTrees.OneHotFeatures[i];
                FeatureBordersDiff(i, feature1.Values, feature2.Values);
            }
        }
     }
     if (model1.ObliviousTrees.CtrFeatures != model2.ObliviousTrees.CtrFeatures) {
        Clog << "CTRFeatures differ" << Endl;
        if (model1.ObliviousTrees.CtrFeatures.size() != model2.ObliviousTrees.CtrFeatures.size()) {
            Clog << "CTRFeatures size differ" << Endl;
        } else {
            for (size_t i = 0; i < model1.ObliviousTrees.CtrFeatures.size(); ++i) {
                auto& feature1 = model1.ObliviousTrees.CtrFeatures[i];
                auto& feature2 = model2.ObliviousTrees.CtrFeatures[i];
                FeatureBordersDiff(i, feature1.Borders, feature2.Borders);
            }
        }
    }
    if (model1.ObliviousTrees != model2.ObliviousTrees) { //TODO(kirillovs): add detailed tree comparator
        Clog << "Oblivious trees differ" << Endl;
    }
    if (model1.ModelInfo != model2.ModelInfo) {
        Clog << "ModelInfo differ" << Endl;
        model1.ModelInfo = THashMap<TString, TString>();
        model2.ModelInfo = THashMap<TString, TString>();
        if (model1 == model2) {
            return 0;
        }
    }
    return 1;
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

