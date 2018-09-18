#include <library/grid_creator/binarization.h>

#include <util/generic/hash_set.h>
#include <util/generic/vector.h>
#include <util/system/types.h>
#include <util/system/unaligned_mem.h>

namespace {
    struct TBestSplitInput {
        TVector<float> Values;
        int BordersCount = 0;
        EBorderSelectionType GridType= EBorderSelectionType::Median;
        bool NanValueIsInfinity = false;
    };

    enum : int {
        MAX_BORDERS_COUNT = 1024
    };

    enum : size_t {
        MAX_VALUES_SIZE = 1ULL * 1024 * 1024,
        MAX_RAM_SIZE    = 4ULL * 1024 * 1024 * 1024
    };
}

static bool TryParse(const int data, EBorderSelectionType* const borderSelectionType) {
    const auto isValid =
        data == (int)EBorderSelectionType::Median
        || data == (int)EBorderSelectionType::GreedyLogSum
        || data == (int)EBorderSelectionType::UniformAndQuantiles
        || data == (int)EBorderSelectionType::MinEntropy
        || data == (int)EBorderSelectionType::MaxLogSum
        || data == (int)EBorderSelectionType::Uniform;
    if (!isValid) {
        return false;
    }

    *borderSelectionType = static_cast<EBorderSelectionType>(data);

    return true;
}

static bool TryParse(const ui8* data, size_t size, TBestSplitInput* const input) {
    if (size < 4) {
        return false;
    }

    input->BordersCount = ReadUnaligned<int>(data);
    data += sizeof(int);
    size -= sizeof(int);

    if (input->BordersCount <= 0 || input->BordersCount > MAX_BORDERS_COUNT) {
        return false;
    }

    if (size < 1) {
        return false;
    }

    if (!TryParse(*data, &input->GridType)) {
        return false;
    }

    data += 1;
    size -= 1;

    if (size < 1) {
        return false;
    }

    input->NanValueIsInfinity = static_cast<bool>(*data);
    data += 1;
    size -= 1;

    const auto valuesSize = size / sizeof(float);
    if (valuesSize > MAX_VALUES_SIZE) {
        return false;
    }

    const auto memoryUseUpperBound = 2ULL * CalcMemoryForFindBestSplit(
        input->BordersCount,
        valuesSize,
        input->GridType);
    if (memoryUseUpperBound > MAX_RAM_SIZE) {
        return false;
    }

    input->Values.resize(valuesSize);
    std::memcpy(input->Values.data(), data, (valuesSize / sizeof(float)) * sizeof(float));

    return true;
}

extern "C" int LLVMFuzzerTestOneInput(const ui8* const data, const size_t size) {
    TBestSplitInput input;
    if (!TryParse(data, size, &input)) {
        return 0;
    }

    BestSplit(input.Values, input.BordersCount, input.GridType, input.NanValueIsInfinity);
    return 0;
}
