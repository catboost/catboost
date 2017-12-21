#include "evaluator.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <fstream>
#include <functional>
#include <iterator>


static const char MODEL_FILE_DESCRIPTOR_CHARS[4] = {'C', 'B', 'M', '1'};

namespace {
    unsigned int GetModelFormatDescriptor() {
        static_assert(sizeof(unsigned int) == 4, "");
        unsigned int result;
        memcpy(&result, MODEL_FILE_DESCRIPTOR_CHARS, sizeof(unsigned int));
        return result;
    }

    template <typename T>
    static inline T Sigmoid(T val) {
        return 1 / (1 + exp(-val));
    }
}

namespace NCatboostStandalone {
    TZeroCopyEvaluator::TZeroCopyEvaluator(const NCatBoostFbs::TModelCore* core)
    {
        SetModelPtr(core);
    }

    double TZeroCopyEvaluator::Apply(
        const std::vector<float>& features,
        EPredictionType predictionType
    ) const {
        std::vector<unsigned char> binaryFeatures(BinaryFeatureCount);
        size_t binFeatureIndex = 0;
        for (const auto& ff : *ObliviousTrees->FloatFeatures()) {
            const auto floatVal = features[ff->Index()];
            for (const auto border : *ff->Borders()) {
                binaryFeatures[binFeatureIndex] = (unsigned char)(floatVal > border);
                ++binFeatureIndex;
            }
        }

        double result = 0.0;
        auto treeSplitsPtr = ObliviousTrees->TreeSplits()->data();
        const auto treeCount =  ObliviousTrees->TreeSizes()->size();
        auto leafValuesPtr = ObliviousTrees->LeafValues()->data();
        for (size_t treeId = 0; treeId < treeCount; ++treeId) {
            const size_t treeSize = ObliviousTrees->TreeSizes()->Get(treeId);
            size_t index{};
            for (size_t depth = 0; depth < treeSize; ++depth) {
                index |= (binaryFeatures[treeSplitsPtr[depth]] << depth);
            }
            result += leafValuesPtr[index];
            treeSplitsPtr += treeSize;
            leafValuesPtr += (1 << treeSize);
        }
        switch(predictionType) {
        case EPredictionType::RawValue:
            return result;
        case EPredictionType::Probability:
            return Sigmoid(result);
        case EPredictionType::Class:
            return result > 0;
        default:
            throw std::runtime_error("unsupported predictionType");
        }
    }

    void TZeroCopyEvaluator::SetModelPtr(const NCatBoostFbs::TModelCore* core) {
        ObliviousTrees = core->ObliviousTrees();
        if (ObliviousTrees == nullptr) {
            throw std::runtime_error(
                "trying to initialize TZeroCopyEvaluator from coreModel without oblivious trees");
        }
        if (ObliviousTrees->CatFeatures() != nullptr && ObliviousTrees->CatFeatures()->size() != 0) {
            throw std::runtime_error(
                "trying to initialize TZeroCopyEvaluator from coreModel with categorical features");
        }
        BinaryFeatureCount = 0;
        FloatFeatureCount = 0;
        for (const auto& ff : *ObliviousTrees->FloatFeatures()) {
            FloatFeatureCount = std::max<int>(FloatFeatureCount, ff->FlatIndex() + 1);
            BinaryFeatureCount += ff->Borders()->size();
        }
    }

    TOwningEvaluator::TOwningEvaluator(const std::string& modelFile) {
        std::ifstream file(modelFile, std::ios::binary);
        ModelBlob.clear();
        ModelBlob.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        InitEvaluator();
    }

    TOwningEvaluator::TOwningEvaluator(std::vector<unsigned char>&& modelBlob) {
        ModelBlob = std::move(modelBlob);
        InitEvaluator();
    }


    TOwningEvaluator::TOwningEvaluator(const std::vector<unsigned char>& modelBlob) {
        ModelBlob = modelBlob;
        InitEvaluator();
    }

    void TOwningEvaluator::InitEvaluator() {
        const auto modelBufferStartOffset = sizeof(unsigned int) * 2;
        if (ModelBlob.empty()) {
            throw std::runtime_error("trying to initialize evaluator from empty ModelBlob");
        }
        {
            const unsigned int* intPtr = reinterpret_cast<const unsigned int*>(ModelBlob.data());
            // verify model file descriptor
            if (intPtr[0] != GetModelFormatDescriptor()) {
                throw std::runtime_error("incorrect model format descriptor");
            }
            // verify model blob length
            if (intPtr[1] + modelBufferStartOffset > ModelBlob.size()) {
                throw std::runtime_error("insufficient model length");
            }
        }
        auto flatbufStartPtr = ModelBlob.data() + modelBufferStartOffset;
        // verify flatbuffers
        {
            flatbuffers::Verifier verifier(flatbufStartPtr, ModelBlob.size() - modelBufferStartOffset);
            if (!NCatBoostFbs::VerifyTModelCoreBuffer(verifier)) {
                throw std::runtime_error("corrupted flatbuffer model");
            }
        }
        auto flatbufModelCore = NCatBoostFbs::GetTModelCore(flatbufStartPtr);
        SetModelPtr(flatbufModelCore);
    }
}

