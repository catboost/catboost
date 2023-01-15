#include "embedding.h"
#include <catboost/private/libs/data_util/path_with_scheme.h>
#include <catboost/private/libs/data_util/line_data_reader.h>

#include <util/generic/string.h>
#include <util/string/split.h>
#include <util/stream/file.h>

namespace NCB {

    inline TVector<float> ConvertToFloat(TConstArrayRef<TString> words) {
        TVector<float> result(words.size());
        for (ui32 i = 0; i < result.size(); ++i) {
            result[i] = FromString<float>(words[i]);
        }
        return result;
    }

    TEmbeddingPtr LoadEmbedding(const TString& path,
                                const TDictionaryProxy& dictionary) {
        const auto delim = '\t';

        TDenseHash<TTokenId, TVector<float>> embeddings;
        const ui32 unknownToken = dictionary.GetUnknownTokenId();

        ui32 embeddingDim = 0;

        TIFStream in(path);
        TString line;
        ui32 lineIdx = 0;
        for (lineIdx = 0; in.ReadLine(line); ++lineIdx) {
            TVector<TString> vals;
            StringSplitter(line).Split(delim).Collect(&vals);
            const auto key = vals[0];
            const auto wordEmbedding = MakeConstArrayRef(vals).Slice(1);

            auto tokenId = dictionary.Apply(key);
            if (tokenId != unknownToken) {
                auto floatEmbedding = ConvertToFloat(wordEmbedding);
                CB_ENSURE(embeddingDim == 0 || embeddingDim == floatEmbedding.size(),
                    "Error: embedding size should be equal for all words. Line #" << lineIdx << ": " << embeddingDim << " ≠ " << floatEmbedding.size());
                embeddingDim = floatEmbedding.size();
                embeddings[TTokenId(tokenId)] = std::move(floatEmbedding);
            }
        }
        Cout << embeddingDim << " " << lineIdx << Endl;

        return CreateEmbedding(std::move(embeddings));
    }
}
