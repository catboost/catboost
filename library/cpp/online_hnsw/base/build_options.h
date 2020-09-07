#pragma once

#include <library/cpp/json/json_reader.h>
#include <util/generic/yexception.h>
#include <util/ysaveload.h>

namespace NOnlineHnsw {
    struct TOnlineHnswBuildOptions {
        size_t MaxNeighbors = 32;
        size_t SearchNeighborhoodSize = 300;
        size_t LevelSizeDecay = AUTO_SELECT;
        size_t NumVertices = AUTO_SELECT;

        static constexpr size_t AUTO_SELECT = 0;

        static TOnlineHnswBuildOptions FromJsonString(const TString& jsonString) {
            NJson::TJsonValue json;
            Y_ENSURE(ReadJsonTree(TStringBuf(jsonString), &json));

            TOnlineHnswBuildOptions options;
            options.MaxNeighbors = json["max_neighbors"].GetUIntegerSafe(options.MaxNeighbors);
            options.SearchNeighborhoodSize = json["search_neighborhood_size"].GetUIntegerSafe(options.SearchNeighborhoodSize);
            options.LevelSizeDecay = json["level_size_decay"].GetUIntegerSafe(options.LevelSizeDecay);
            options.NumVertices = json["num_vertices"].GetUIntegerSafe(options.NumVertices);

            return options;
        }

        void CheckOptions() const {
            Y_ENSURE(1 <= MaxNeighbors && MaxNeighbors <= SearchNeighborhoodSize);
            Y_ENSURE(LevelSizeDecay == AUTO_SELECT || LevelSizeDecay > 1);
        }

        Y_SAVELOAD_DEFINE(
            MaxNeighbors,
            SearchNeighborhoodSize,
            LevelSizeDecay,
            NumVertices);
    };
} // namespace NOnlineHnsw
