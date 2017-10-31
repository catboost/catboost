#pragma once

#include <catboost/libs/model/model.h>
#include <catboost/libs/model/ctr_provider.h>
#include <catboost/libs/cat_feature/cat_feature.h>
#include <catboost/libs/helpers/exception.h>
#include <util/system/hp_timer.h>

namespace NCatBoost {
 /**
 * Class for fast model apply:
 * Uses binarized feature values for fast tree index calculations
 * Binarized features in worst case consumes N_trees * max(TreeDepth) bytes
 * WARNING: Currently it builds optimized tree structures on construction or model load, try reuse calcer
 */
    class TFormulaEvaluator {
        class TFeatureIndexProvider: public IFeatureIndexProvider {
        public:
            TFeatureIndexProvider(const TFormulaEvaluator& calcer)
                : Calcer(calcer)
            {
            }
            int GetBinFeatureIdx(const TBinFeature& feature) const override {
                return Calcer.BinFeatureIndexes.at(TModelSplit(feature));
            }
            int GetBinFeatureIdx(const TOneHotFeature& feature) const override {
                return Calcer.BinFeatureIndexes.at(TModelSplit(feature));
            }
        private:
            const TFormulaEvaluator& Calcer;
        };
        friend class TFeatureIndexProvider;

    public:
        TFormulaEvaluator() = default;
        TFormulaEvaluator(TFullModel&& model) {
            InitFromFullModel(std::move(model));
        }
        TFormulaEvaluator(const TFullModel& model) {
            InitFromFullModel(model);
        }

        template<typename TFloatFeatureAccessor, typename TCatFeatureAccessor>
        void CalcGeneric(TFloatFeatureAccessor floatFeatureAccessor, TCatFeatureAccessor catFeaturesAccessor, size_t docCount, size_t treeStart, size_t treeEnd, TArrayRef<double> results) const {
            size_t blockSize;
            if (UsedCtrFeatures.empty()) {
                blockSize = 128;
            } else {
                blockSize = 4096;
            }
            blockSize = Min(blockSize, docCount);
            CB_ENSURE(results.size() == docCount * ModelClassCount);
            std::fill(results.begin(), results.end(), 0.0);
            yvector<ui8> binFeatures(UsedBinaryFeaturesCount * blockSize);
            yvector<ui32> indexesVec(blockSize);
            yvector<int> transposedHash(blockSize * CatFeatureFlatIndex.size());
            yvector<float> ctrs(UsedModelCtrs.size() * blockSize);
            for (size_t blockStart = 0; blockStart < docCount; blockStart += blockSize) {
                const auto docCountInBlock = Min(blockSize, docCount - blockStart);
                BinarizeFeatures(floatFeatureAccessor, catFeaturesAccessor, blockStart, blockStart + docCountInBlock, binFeatures, transposedHash, ctrs);
                CalcTrees(blockStart, binFeatures, docCountInBlock, indexesVec, treeStart, treeEnd, results);
            }
        }

        template<typename TFloatFeatureAccessor, typename TCatFeatureAccessor>
        yvector<yvector<double>> CalcTreeIntervalsGeneric(TFloatFeatureAccessor floatFeatureAccessor, TCatFeatureAccessor catFeaturesAccessor, size_t docCount, size_t incrementStep) const {
            size_t blockSize;
            if (UsedCtrFeatures.empty()) {
                blockSize = 128;
            } else {
                blockSize = 4096;
            }
            blockSize = Min(blockSize, docCount);
            auto treeStepCount = (BinaryTrees.size() + incrementStep - 1) / incrementStep;
            yvector<yvector<double>> results(docCount, yvector<double>(treeStepCount));
            CB_ENSURE(ModelClassCount == 1);
            yvector<ui8> binFeatures(UsedBinaryFeaturesCount * blockSize);
            yvector<ui32> indexesVec(blockSize);
            yvector<int> transposedHash(blockSize * CatFeatureFlatIndex.size());
            yvector<float> ctrs(UsedModelCtrs.size() * blockSize);
            yvector<double> tmpResult(docCount);
            TArrayRef<double> tmpResultRef(tmpResult);
            for (size_t blockStart = 0; blockStart < docCount; blockStart += blockSize) {
                const auto docCountInBlock = Min(blockSize, docCount - blockStart);
                BinarizeFeatures(floatFeatureAccessor, catFeaturesAccessor, blockStart, blockStart + docCountInBlock, binFeatures, transposedHash, ctrs);
                for (size_t stepIdx = 0; stepIdx < treeStepCount; ++stepIdx) {
                    CalcTrees(blockStart,
                              binFeatures,
                              docCountInBlock,
                              indexesVec,
                              stepIdx * incrementStep,
                              Min((stepIdx + 1) * incrementStep, BinaryTrees.size()),
                              tmpResultRef);
                    for (size_t i = 0; i < docCountInBlock; ++i) {
                        results[blockStart + i][stepIdx] = tmpResult[i];
                    }
                }
            }
            return results;
        }

        void CalcFlatTransposed(const yvector<TConstArrayRef<float>>& transposedFeatures, size_t treeStart, size_t treeEnd, TArrayRef<double> results) const;
        void CalcFlat(const yvector<TConstArrayRef<float>>& features, size_t treeStart, size_t treeEnd, TArrayRef<double> results) const;
        void CalcFlat(const yvector<TConstArrayRef<float>>& features, TArrayRef<double> results) const {
            CalcFlat(features, 0, BinaryTrees.size(), results);
        }
        void CalcFlat(TConstArrayRef<float> features, TArrayRef<double> result) const {
            yvector<TConstArrayRef<float>> featuresVec = {features};
            CalcFlat(featuresVec, result);
        }

        yvector<yvector<double>> CalcTreeIntervals(
            const yvector<TConstArrayRef<float>>& floatFeatures,
            const yvector<TConstArrayRef<int>>& catFeatures,
            size_t incrementStep) const;

        yvector<yvector<double>> CalcTreeIntervalsFlat(
            const yvector<TConstArrayRef<float>>& mixedFeatures,
            size_t incrementStep) const;

        void Calc(const yvector<TConstArrayRef<float>>& floatFeatures,
                  const yvector<TConstArrayRef<int>>& catFeatures,
                  size_t treeStart,
                  size_t treeEnd,
                  TArrayRef<double> results) const;
        void Calc(const yvector<TConstArrayRef<float>>& floatFeatures,
                  const yvector<TConstArrayRef<int>>& catFeatures,
                  TArrayRef<double> results) const {
            Calc(floatFeatures, catFeatures, 0, BinaryTrees.size(), results);
        }

        void Calc(TConstArrayRef<float> floatFeatures,
                    TConstArrayRef<int> catFeatures,
                  TArrayRef<double> result) const {
            yvector<TConstArrayRef<float>> floatFeaturesVec = {floatFeatures};
            yvector<TConstArrayRef<int>> catFeaturesVec = {catFeatures};
            Calc(floatFeaturesVec, catFeaturesVec, result);
        }

        void Calc(const yvector<TConstArrayRef<float>>& floatFeatures,
                  const yvector<yvector<TStringBuf>>& catFeatures,
                  size_t treeStart,
                  size_t treeEnd,
                  TArrayRef<double> results) const;

        void Calc(const yvector<TConstArrayRef<float>>& floatFeatures,
                  const yvector<yvector<TStringBuf>>& catFeatures,
                  TArrayRef<double> results) const {
            Calc(floatFeatures, catFeatures, 0, BinaryTrees.size(), results);
        }



        void InitFromCoreModel(const TCoreModel& coreModel) {
            InitBinTreesFromCoreModel(coreModel);
            CtrProvider.Reset();
        }
        void InitFromFullModel(const TFullModel& fullModel);
        void InitFromFullModel(TFullModel&& fullModel);

        size_t GetFloatFeaturesUsed() const {
            return FloatFeaturesCount;
        }

        size_t GetCatFeaturesUsed() const {
            return CatFeaturesCount;
        }

        size_t GetFlatFeatureVectorExpectedSize() const {
            return CatFeaturesCount + FloatFeaturesCount;
        }

        size_t GetBinFeaturesCount() const {
            return UsedBinaryFeaturesCount;
        }
        size_t GetModelClassCount() const {
            return ModelClassCount;
        }
        int GetTreeCount() const {
            return static_cast<int>(BinaryTrees.size());
        }

        bool HasCategoricalFeatures() const {
            return CatFeaturesCount > 0;
        }
        // if no ctr features present it'll return false
        bool HasValidCtrProvider() const {
            if (!CtrProvider) {
                return false;
            }
            return CtrProvider->HasNeededCtrs(UsedModelCtrs);
        }

        TFormulaEvaluator CopyTreeRange(const size_t begin, const size_t end) const;

        void TruncateModel(const size_t begin, const size_t end);

    private:
        void InitBinTreesFromCoreModel(const TCoreModel& model);

        void OneHotBinsFromTransposedCatFeatures(
            const size_t docCount,
            yvector<ui8>& result,
            yvector<int>& transposedHash,
            size_t& currentBinIndex) const {
            for (const auto& oheFeature : UsedOHEFeatures) {
                const auto catIdx = oheFeature.CatFeatureIndex;
                for (size_t docId = 0; docId < docCount; ++docId) {
                    const auto val = transposedHash[catIdx * docCount + docId];
                    auto writeIdx = docId + currentBinIndex;
                    for (size_t borderIdx = 0; borderIdx < oheFeature.Values.size(); ++borderIdx) {
                        result[writeIdx] = (ui8)(val == oheFeature.Values[borderIdx]);
                        writeIdx += docCount;
                    }
                }
                currentBinIndex += oheFeature.Values.size() * docCount;
            }
        }

        void BinarizeFloatCtrs(const size_t docCount, size_t currentBinIndex, const yvector<float>& ctrs, yvector<ui8>& result) const {
            const auto docCount4 = (docCount | 0x3) ^0x3;
            for (size_t i = 0; i < UsedCtrFeatures.size(); ++i) {
                const auto& ctr = UsedCtrFeatures[i];
                auto ctrFloatsPtr = &ctrs[i * docCount];
                for (size_t docId = 0; docId < docCount4; docId += 4) {
                    const float val[4] = {
                        ctrFloatsPtr[docId + 0],
                        ctrFloatsPtr[docId + 1],
                        ctrFloatsPtr[docId + 2],
                        ctrFloatsPtr[docId + 3]
                    };

                    auto writePtr = &result[docId + currentBinIndex];
                    for (const auto border : ctr.Borders) {
                        writePtr[0] = (ui8)(val[0] > border);
                        writePtr[1] = (ui8)(val[1] > border);
                        writePtr[2] = (ui8)(val[2] > border);
                        writePtr[3] = (ui8)(val[3] > border);
                        writePtr += docCount;
                    }
                }
                for (size_t docId = docCount4; docId < docCount; ++docId) {
                    const auto val = ctrFloatsPtr[docId];
                    auto writePtr = &result[docId + currentBinIndex];
                    for (const auto border : ctr.Borders) {
                        *writePtr = (ui8)(val > border);
                        writePtr += docCount;
                    }
                }
                currentBinIndex += ctr.Borders.size() * docCount;
            }
        }

        template<typename TFloatFeatureAccessor, typename TCatFeatureAccessor>
        void BinarizeFeatures(
            TFloatFeatureAccessor floatAccessor,
            TCatFeatureAccessor catFeatureAccessor,
            size_t start,
            size_t end,
            yvector<ui8>& result,
            yvector<int>& transposedHash,
            yvector<float>& ctrs
            ) const {
            const auto docCount = end - start;
            const auto docCount4 = (docCount | 0x3) ^ 0x3;
            size_t currentBinIndex = 0;
            for (const auto& floatFeature : UsedFloatFeatures) {
                for (size_t docId = 0; docId < docCount4; docId += 4) {
                    const float val[4] =
                    {
                        floatAccessor(floatFeature, start + docId + 0),
                        floatAccessor(floatFeature, start + docId + 1),
                        floatAccessor(floatFeature, start + docId + 2),
                        floatAccessor(floatFeature, start + docId + 3)
                    };
                    auto writePtr = &result[docId + currentBinIndex];
                    for (const auto border : floatFeature.Borders) {
                        writePtr[0] = (ui8)(val[0] > border);
                        writePtr[1] = (ui8)(val[1] > border);
                        writePtr[2] = (ui8)(val[2] > border);
                        writePtr[3] = (ui8)(val[3] > border);
                        writePtr += docCount;
                    }
                }
                for (size_t docId = docCount4; docId < docCount; ++docId) {
                    const auto val = floatAccessor(floatFeature, start + docId);
                    auto writePtr = &result[docId + currentBinIndex];
                    for (const auto border : floatFeature.Borders) {
                        *writePtr = (ui8)(val > border);
                        writePtr += docCount;
                    }
                }
                currentBinIndex += floatFeature.Borders.size() * docCount;
            }
            auto catFeatureCount = CatFeatureFlatIndex.size();
            if (catFeatureCount > 0) {
                for (size_t docId = 0; docId < docCount; ++docId) {
                    auto idx = docId;
                    for (size_t i = 0; i < catFeatureCount; ++i) {
                        transposedHash[idx] = catFeatureAccessor(i, start + docId);
                        idx += docCount;
                    }
                }
                OneHotBinsFromTransposedCatFeatures(docCount, result, transposedHash, currentBinIndex);
                CtrProvider->CalcCtrs(UsedModelCtrs,
                                      result,
                                      transposedHash,
                                      TFeatureIndexProvider(*this),
                                      docCount,
                                      ctrs);
                BinarizeFloatCtrs(docCount, currentBinIndex, ctrs, result);
            }
        }

        void CalcTrees(size_t blockStart,
                       const yvector<ui8>& binFeatures,
                       const unsigned long docCountInBlock,
                       yvector<ui32>& indexesVec,
                       size_t treeStart,
                       size_t treeEnd,
                       TArrayRef<double>& results) const {
            const auto docCountInBlock4 = (docCountInBlock | 0x3) ^0x3;
            for (size_t treeId = treeStart; treeId < treeEnd; ++treeId) {
                memset(indexesVec.data(), 0, sizeof(ui32) * docCountInBlock);
                const auto& tree = BinaryTrees[treeId];
                for (int depth = 0; depth < tree.ysize(); ++depth) {
                    auto indexesPtr = indexesVec.data();
                    const auto bin = tree[depth];
                    auto binFeaturePtr = &binFeatures[bin * docCountInBlock];
                    for (size_t docId = 0; docId < docCountInBlock; ++docId) {
                        indexesPtr[docId] |= binFeaturePtr[docId] << depth;
                    }
                }
                auto treeLeafPtr = LeafValues[treeId].data();
                if (ModelClassCount == 1) { // single class model
                    auto indexesPtr = indexesVec.data();
                    auto writePtr = &results[blockStart];
                    for (size_t docId = 0; docId < docCountInBlock4; docId += 4) {
                        writePtr[0] += treeLeafPtr[indexesPtr[0]];
                        writePtr[1] += treeLeafPtr[indexesPtr[1]];
                        writePtr[2] += treeLeafPtr[indexesPtr[2]];
                        writePtr[3] += treeLeafPtr[indexesPtr[3]];
                        writePtr += 4;
                        indexesPtr += 4;
                    }
                    for (size_t docId = docCountInBlock4; docId < docCountInBlock; ++docId) {
                        *writePtr += treeLeafPtr[*indexesPtr];
                        ++writePtr;
                        ++indexesPtr;
                    }
                } else { // mutliclass model
                    auto indexesPtr = indexesVec.data();
                    auto docResultPtr = &results[blockStart * ModelClassCount];
                    for (size_t docId = 0; docId < docCountInBlock; ++docId) {
                        auto leafValuePtr = treeLeafPtr + indexesPtr[docId] * ModelClassCount;
                        for (size_t classId = 0; classId < ModelClassCount; ++classId) {
                            docResultPtr[classId] += leafValuePtr[classId];
                        }
                        docResultPtr += ModelClassCount;
                    }
                }
            }
        }
    protected:
        struct TFloatFeature {
            int FeatureIndex = -1;
            yvector<float> Borders;
        };
        struct TOHEFeature {
            int CatFeatureIndex = -1;
            yvector<int> Values;
        };
        struct TCtrFeature {
            TModelCtr Ctr;
            yvector<float> Borders;
        };
        yhash<TModelSplit, int> BinFeatureIndexes;
        yvector<int> CatFeatureFlatIndex;
        yvector<int> FloatFeatureFlatIndex;

        yvector<TFloatFeature> UsedFloatFeatures;
        yvector<TOHEFeature> UsedOHEFeatures;
        yvector<TCtrFeature> UsedCtrFeatures;
        // contains unpacked unique TModelCtr from UsedCtrFeatures, duplicated for ctr calculation speed
        yvector<TModelCtr> UsedModelCtrs;
        size_t UsedBinaryFeaturesCount = 0;
        // ApproxDimension from TFullModel
        size_t ModelClassCount = 0;

        size_t FloatFeaturesCount = 0;
        size_t CatFeaturesCount = 0;

        // oblivious bin features trees
        yvector<yvector<int>> BinaryTrees;
        yvector<yvector<double>> LeafValues; // [numTree][bucketId * classCount + classId]
        TIntrusivePtr<ICtrProvider> CtrProvider;
    };
}
