#include "pairs_data_loaders.h"

#include "visitor.h"

#include <catboost/private/libs/data_util/exists_checker.h>
#include <catboost/private/libs/data_util/line_data_reader.h>

#include <catboost/libs/helpers/exception.h>

#include <util/generic/hash.h>
#include <util/generic/strbuf.h>
#include <util/generic/vector.h>
#include <util/string/cast.h>
#include <util/string/split.h>
#include <util/system/compiler.h>


namespace NCB {
    void IPairsDataLoader::SetGroupIdToIdxMap(const THashMap<TGroupId, ui32>* groupIdToIdxMap) {
        Y_UNUSED(groupIdToIdxMap);
        CB_ENSURE_INTERNAL(
            false,
            "IPairsDataLoader::SetGroupIdToIdxMap called for loader that does not need groupIdToIdxMap"
        );
    }

    void NCB::TDsvFlatPairsLoader::Do(IDatasetVisitor* visitor) {
        THolder<ILineDataReader> reader = GetLineDataReader(Args.Path);

        TVector<TPair> pairs;
        TString line;
        for (size_t lineNumber = 0; reader->ReadLine(&line); lineNumber++) {
            TVector<TString> tokens = StringSplitter(line).Split('\t');
            if (tokens.empty()) {
                continue;
            }
            try {
                CB_ENSURE(tokens.ysize() == 2 || tokens.ysize() == 3,
                    "Each line should have two or three columns. This line has " << tokens.size()
                );
                TPair pair;

                size_t tokenIdx = 0;
                auto parseIdFunc = [&](TStringBuf description, ui32* id) {
                    CB_ENSURE(
                        TryFromString(tokens[tokenIdx], *id),
                        "Invalid " << description << " index: cannot parse as nonnegative index ("
                        << tokens[tokenIdx] << ')'
                    );
                    *id -= Args.DatasetSubset.Range.Begin;
                    ++tokenIdx;
                };
                parseIdFunc(AsStringBuf("Winner"), &pair.WinnerId);
                parseIdFunc(AsStringBuf("Loser"), &pair.LoserId);

                if (tokens.ysize() == 3) {
                    CB_ENSURE(
                        TryFromString(tokens[2], pair.Weight),
                        "Invalid weight: cannot parse as float (" << tokens[2] << ')'
                    );
                } else {
                    pair.Weight = 1.0f;
                }
                if (pair.WinnerId < Args.DatasetSubset.GetSize() &&
                    pair.LoserId < Args.DatasetSubset.GetSize())
                {
                    pairs.push_back(std::move(pair));
                } else {
                    CB_ENSURE(
                        pair.WinnerId >= Args.DatasetSubset.GetSize() &&
                        pair.LoserId >= Args.DatasetSubset.GetSize(),
                        "Load subset " << Args.DatasetSubset.Range << " must contain loser "
                        << pair.LoserId + Args.DatasetSubset.Range.Begin << " and winner "
                        << pair.WinnerId + Args.DatasetSubset.Range.Begin
                    );
                }
            } catch (const TCatBoostException& e) {
                throw TCatBoostException() << "Incorrect pairs data. Invalid line number #"
                    << lineNumber << ": " << e.what();
            }
        }

        visitor->SetPairs(TRawPairsData(std::move(pairs)));
    }


    class TDsvGroupedPairsLoader : public IPairsDataLoader {
    public:
        explicit TDsvGroupedPairsLoader(TPairsDataLoaderArgs&& args)
            : Args(std::move(args))
        {}

        bool NeedGroupIdToIdxMap() const override { return true; }
        void SetGroupIdToIdxMap(const THashMap<TGroupId, ui32>* groupIdToIdxMap) override {
            GroupIdToIdxMap = groupIdToIdxMap;
        }

        void Do(IDatasetVisitor* visitor) override {
            CB_ENSURE_INTERNAL(GroupIdToIdxMap, "GroupIdToIdxMap has not been initialized");

            THolder<ILineDataReader> reader = GetLineDataReader(Args.Path);

            TVector<TPairInGroup> pairs;
            TString line;
            for (size_t lineNumber = 0; reader->ReadLine(&line); lineNumber++) {
                TVector<TString> tokens = StringSplitter(line).Split('\t');
                if (tokens.empty()) {
                    continue;
                }
                try {
                    CB_ENSURE(tokens.ysize() == 3 || tokens.ysize() == 4,
                        "Each line should have two or three columns. This line has " << tokens.size()
                    );
                    TPairInGroup pair;

                    TGroupId groupId = CalcGroupIdFor(tokens[0]);
                    auto groupIdxIt = GroupIdToIdxMap->find(groupId);
                    if (groupIdxIt == GroupIdToIdxMap->end()) {
                        continue;
                    }
                    pair.GroupIdx = groupIdxIt->second;

                    size_t tokenIdx = 1;
                    auto parseIdFunc = [&](TStringBuf description, ui32* id) {
                        CB_ENSURE(
                            TryFromString(tokens[tokenIdx], *id),
                            "Invalid " << description << " index: cannot parse as nonnegative index ("
                            << tokens[tokenIdx] << ')'
                        );
                        ++tokenIdx;
                    };
                    parseIdFunc(AsStringBuf("WinnerIdxInGroup"), &pair.WinnerIdxInGroup);
                    parseIdFunc(AsStringBuf("LoserIdxInGroup"), &pair.LoserIdxInGroup);

                    if (tokens.ysize() == 4) {
                        CB_ENSURE(
                            TryFromString(tokens[3], pair.Weight),
                            "Invalid weight: cannot parse as float (" << tokens[3] << ')'
                        );
                    } else {
                        pair.Weight = 1.0f;
                    }

                    pairs.push_back(std::move(pair));
                } catch (const TCatBoostException& e) {
                    throw TCatBoostException() << "Incorrect pairs data. Invalid line number #"
                        << lineNumber << ": " << e.what();
                }
            }

            visitor->SetPairs(TRawPairsData(std::move(pairs)));
        }

    private:
        TPairsDataLoaderArgs Args;
        const THashMap<TGroupId, ui32>* GroupIdToIdxMap = nullptr;
    };

    namespace {
        TExistsCheckerFactory::TRegistrator<TFSExistsChecker> DsvFlatExistsCheckerReg("dsv-flat");
        TLineDataReaderFactory::TRegistrator<TFileLineDataReader> DsvFlatLineDataReaderReg("dsv-flat");
        TPairsDataLoaderFactory::TRegistrator<TDsvFlatPairsLoader> DsvFlatPairsDataLoaderReg("dsv-flat");

        TExistsCheckerFactory::TRegistrator<TFSExistsChecker> DsvGroupedExistsCheckerReg("dsv-grouped");
        TLineDataReaderFactory::TRegistrator<TFileLineDataReader> DsvGroupedLineDataReaderReg("dsv-grouped");
        TPairsDataLoaderFactory::TRegistrator<TDsvGroupedPairsLoader> DsvGroupedPairsDataLoaderReg("dsv-grouped");
    }

}

