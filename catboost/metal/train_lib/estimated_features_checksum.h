#pragma once

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/data/objects.h>
#include <catboost/libs/helpers/checksum.h>
#include <catboost/libs/helpers/exception.h>

namespace NCB {

// Fingerprint the source representation consumed by the shared estimators.
// Tokenized text is already a dictionary-relative bag of tokens here, just as
// numeric snapshot checksums use quantized bins rather than original floats.
// Estimator/dictionary GUIDs are deliberately excluded: they are regenerated
// when an equivalent Pool is loaded again.
inline ui32 UpdateMetalEstimatedSourceChecksum(
    ui32 checksum,
    const TTrainingDataProviders& data,
    NPar::ILocalExecutor* executor)
{
    const auto hasEstimatedSource = [](const TTrainingDataProvider& dataset) {
        const auto& layout = *dataset.ObjectsData->GetFeaturesLayout();
        for (const auto type : {EFeatureType::Text, EFeatureType::Embedding}) {
            for (ui32 feature = 0; feature < layout.GetFeatureCount(type); ++feature) {
                if (layout.GetInternalFeatureMetaInfo(feature, type).IsAvailable) {
                    return true;
                }
            }
        }
        return false;
    };

    bool hasSources = hasEstimatedSource(*data.Learn);
    for (const auto& test : data.Test) {
        hasSources |= hasEstimatedSource(*test);
    }
    // Preserve existing numeric/categorical snapshot identities byte for byte.
    if (!hasSources) {
        return checksum;
    }

    checksum = UpdateCheckSum(checksum, TStringBuf("MetalEstimatedSourcesV1"),
        static_cast<ui64>(data.Test.size()));
    const auto addDataset = [&](const TTrainingDataProvider& dataset, ui64 datasetIndex) {
        const auto& objects = *dataset.ObjectsData;
        const auto& layout = *objects.GetFeaturesLayout();
        const ui32 rows = dataset.GetObjectCount();
        checksum = UpdateCheckSum(checksum, datasetIndex, rows);
        for (const auto type : {EFeatureType::Text, EFeatureType::Embedding}) {
            const ui32 featureCount = layout.GetFeatureCount(type);
            checksum = UpdateCheckSum(checksum, static_cast<ui32>(type), featureCount);
            for (ui32 feature = 0; feature < featureCount; ++feature) {
                const auto& meta = layout.GetInternalFeatureMetaInfo(feature, type);
                checksum = UpdateCheckSum(checksum, feature,
                    layout.GetExternalFeatureIdx(feature, type), meta.IsAvailable,
                    static_cast<ui64>(meta.Name.size()), TStringBuf(meta.Name));
                if (!meta.IsAvailable) {
                    continue;
                }
                if (type == EFeatureType::Text) {
                    const auto holder = objects.GetTextFeature(feature);
                    CB_ENSURE(holder, "Snapshot tokenized text feature values are missing");
                    const auto values = (*holder)->ExtractValues(executor);
                    CB_ENSURE(values.GetSize() == rows, "Snapshot tokenized text row count differs");
                    for (const auto& text : values) {
                        ui64 tokenCount = 0;
                        for (const auto& token : text) {
                            checksum = UpdateCheckSum(checksum,
                                static_cast<ui32>(token.Token()), token.Count());
                            ++tokenCount;
                        }
                        // Include row boundaries: concatenating token pairs
                        // alone would confuse empty rows and shifted tokens.
                        checksum = UpdateCheckSum(checksum, tokenCount);
                    }
                } else {
                    const auto holder = objects.GetEmbeddingFeature(feature);
                    CB_ENSURE(holder, "Snapshot embedding feature values are missing");
                    const auto values = (*holder)->ExtractValues(executor);
                    CB_ENSURE(values.GetSize() == rows, "Snapshot embedding row count differs");
                    for (const auto& embedding : values) {
                        checksum = UpdateCheckSum(checksum,
                            static_cast<ui64>(embedding.GetSize()), *embedding);
                    }
                }
            }
        }
    };
    addDataset(*data.Learn, 0);
    for (ui64 test = 0; test < data.Test.size(); ++test) {
        addDataset(*data.Test[test], test + 1);
    }
    return checksum;
}

} // namespace NCB
