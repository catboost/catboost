#include "connected_components.h"

namespace NCB {

    namespace {
        struct TDocument {
            ui32 Id;
            bool Used = false;
            TVector<TDocument*> Competitors;

            explicit TDocument(ui32 Id)
                : Id(Id)
                , Used(false)
            {}
        };
    }

    static void DepthFirstSearch(
        TDocument* currentDocument,
        TVector<ui32>* permutationForGrouping,
        TVector<ui32>* newPositions
    ) {
        currentDocument->Used = true;
        (*newPositions)[currentDocument->Id] = permutationForGrouping->size();
        permutationForGrouping->emplace_back(currentDocument->Id);
        for (const auto& newDocument : currentDocument->Competitors) {
            if (!newDocument->Used) {
                DepthFirstSearch(newDocument, permutationForGrouping, newPositions);
            }
        }
    }

    void ConstructConnectedComponents(
        ui32 docCount,
        const TConstArrayRef<TPair> pairs,
        TVector<ui32>* groupBounds,
        TVector<ui32>* permutationForGrouping,
        TVector<TPair>* pairsInPermutedDataset
    ) {
        TVector<TDocument> documents;
        documents.reserve(docCount);
        for (ui32 i = 0; i < docCount; ++i) {
            documents.emplace_back(i);
        }
        for (const auto& pair : pairs) {
            documents[pair.WinnerId].Competitors.emplace_back(&documents[pair.LoserId]);
            documents[pair.LoserId].Competitors.emplace_back(&documents[pair.WinnerId]);
        }
        TVector<ui32> newPositions(docCount);
        permutationForGrouping->reserve(docCount);
        for (ui32 i = 0; i < docCount; ++i) {
            if (!documents[i].Used) {
                DepthFirstSearch(&documents[i], permutationForGrouping, &newPositions);
                groupBounds->emplace_back(permutationForGrouping->size());
            }
        }
        pairsInPermutedDataset->reserve(pairs.size());
        for (const auto& pair : pairs) {
            pairsInPermutedDataset->emplace_back(newPositions[pair.WinnerId], newPositions[pair.LoserId], pair.Weight);
        }
    }
}
