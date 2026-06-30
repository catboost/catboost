#include "pairs_data_loaders.h"

#include "visitor.h"

#include <catboost/private/libs/data_util/exists_checker.h>
#include <catboost/private/libs/data_util/line_data_reader.h>

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/logging/logging.h>

#include <util/generic/hash.h>
#include <util/generic/strbuf.h>
#include <util/generic/vector.h>
#include <util/generic/ymath.h>
#include <util/string/cast.h>
#include <util/string/split.h>
#include <util/system/compiler.h>
#include <util/system/hp_timer.h>

namespace NCB {
    void IPairsDataLoader::SetGroupIdToIdxMap(TConstArrayRef<TGroupId>) {
        CB_ENSURE_INTERNAL(
            false,
            "IPairsDataLoader::SetGroupIdToIdxMap called for loader that does not need groupIdToIdxMap"
        );
    }

    void NCB::TDsvFlatPairsLoader::Do(IDatasetVisitor* visitor) {
        THolder<ILineDataReader> reader = GetLineDataReader(Args.Path, NCB::TDsvFormatOptions(), /*keepLineOrder*/false);

        const auto approxmatePairsCount = reader->GetDataLineCount(/*estimate*/true);
        TVector<TPair> pairs;
        pairs.reserve(approxmatePairsCount);
        TString line;
        ui64 lineNumber;
        THPTimer progressTimer;
        ui64 progressIndex = 0;
        while (reader->ReadLine(&line, &lineNumber)) {
            if (progressTimer.Passed() > 60/*seconds*/) {
                if (progressIndex < approxmatePairsCount) {
                    CATBOOST_DEBUG_LOG << "Last minute status: " << progressIndex / CeilDiv<ui32>(approxmatePairsCount, 100) << "% pairs loaded" << Endl;
                } else {
                    CATBOOST_DEBUG_LOG << "Last minute status: " << progressIndex << " pairs loaded" << Endl;
                }
                progressTimer.Reset();
            }
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
                parseIdFunc(TStringBuf("Winner"), &pair.WinnerId);
                parseIdFunc(TStringBuf("Loser"), &pair.LoserId);

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
            ++progressIndex;
        }

        if (IsPairs) {
            visitor->SetPairs(TRawPairsData(std::move(pairs)));
        } else {
            visitor->SetGraph(TRawPairsData(std::move(pairs)));
        }
    }


    THashMap<TGroupId, ui32> ConvertGroupIdToIdxMap(TConstArrayRef<TGroupId> groupIdsArray) {
        THashMap<TGroupId, ui32> groupIdToIdxMap;
        if (!groupIdsArray.empty()) {
            TGroupId currentGroupId = groupIdsArray[0];
            ui32 currentGroupIdx = 0;

            auto insertCurrentGroup = [&] () {
                CB_ENSURE(
                        !groupIdToIdxMap.contains(currentGroupId),
                        "Group id " << currentGroupId << " is used for several groups in the dataset"
                );
                groupIdToIdxMap.emplace(currentGroupId, currentGroupIdx++);
            };

            for (TGroupId groupId : groupIdsArray) {
                if (groupId != currentGroupId) {
                    insertCurrentGroup();
                    currentGroupId = groupId;
                }
            }
            insertCurrentGroup();
        }
        return groupIdToIdxMap;
    }


    class TDsvGroupedPairsLoader : public IPairsDataLoader {
    public:
        explicit TDsvGroupedPairsLoader(TPairsDataLoaderArgs&& args)
            : Args(std::move(args))
        {}

        bool NeedGroupIdToIdxMap() const override {
            return Args.Path.Scheme != ("dsv-grouped-with-idx");
        }
        void SetGroupIdToIdxMap(TConstArrayRef<TGroupId> groupIdsArray) override {
            GroupIdToIdxMap = std::move(ConvertGroupIdToIdxMap(groupIdsArray));
        }

        void Do(IDatasetVisitor* visitor) override {
            // callback returns true if this pair should be added to the result
            std::function<bool(TStringBuf, ui32*)> calcGroupIdxCallback;

            if (Args.Path.Scheme == ("dsv-grouped-with-idx")) {
                CB_ENSURE(
                    Args.DatasetSubset.Range.End == Max<ui64>(),
                    "Pairs Scheme 'dsv-grouped-with-idx' does not support loading of subsets"
                );

                calcGroupIdxCallback = [] (TStringBuf token, ui32* groupIdx) -> bool {
                    CB_ENSURE(
                        TryFromString(token, *groupIdx),
                        "Cannot parse string ("
                        << token << ") and a groupIdx"
                    );
                    return true;
                };
            } else {
                CB_ENSURE_INTERNAL(GroupIdToIdxMap, "GroupIdToIdxMap has not been initialized");

                calcGroupIdxCallback = [&] (TStringBuf token, ui32* groupIdx) -> bool {
                    TGroupId groupId = CalcGroupIdFor(token);
                    auto groupIdxIt = GroupIdToIdxMap->find(groupId);
                    if (groupIdxIt == GroupIdToIdxMap->end()) {
                        return false;
                    }
                    *groupIdx = groupIdxIt->second;
                    return true;
                };
            }

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

                    if (!calcGroupIdxCallback(tokens[0], &pair.GroupIdx)) {
                        continue;
                    }

                    size_t tokenIdx = 1;
                    auto parseIdFunc = [&](TStringBuf description, ui32* id) {
                        CB_ENSURE(
                            TryFromString(tokens[tokenIdx], *id),
                            "Invalid " << description << " index: cannot parse as nonnegative index ("
                            << tokens[tokenIdx] << ')'
                        );
                        ++tokenIdx;
                    };
                    parseIdFunc(TStringBuf("WinnerIdxInGroup"), &pair.WinnerIdxInGroup);
                    parseIdFunc(TStringBuf("LoserIdxInGroup"), &pair.LoserIdxInGroup);

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

            if (IsPairs) {
                visitor->SetPairs(TRawPairsData(std::move(pairs)));
            } else {
                visitor->SetGraph(TRawPairsData(std::move(pairs)));
            }
        }

    private:
        TPairsDataLoaderArgs Args;
        TMaybe<THashMap<TGroupId, ui32>> GroupIdToIdxMap;
    };

    namespace {
        // lines are (flatWinnerIdx, flatLoserIdx, [opt] weight)
        TExistsCheckerFactory::TRegistrator<TFSExistsChecker> DsvFlatExistsCheckerReg("dsv-flat");
        TLineDataReaderFactory::TRegistrator<TFileLineDataReader> DsvFlatLineDataReaderReg("dsv-flat");
        TPairsDataLoaderFactory::TRegistrator<TDsvFlatPairsLoader> DsvFlatPairsDataLoaderReg("dsv-flat");

        // lines are (groupId as string, winnerIdxInGroup, loserIdxInGroup, [opt] weight)
        TExistsCheckerFactory::TRegistrator<TFSExistsChecker> DsvGroupedExistsCheckerReg("dsv-grouped");
        TLineDataReaderFactory::TRegistrator<TFileLineDataReader> DsvGroupedLineDataReaderReg("dsv-grouped");
        TPairsDataLoaderFactory::TRegistrator<TDsvGroupedPairsLoader> DsvGroupedPairsDataLoaderReg("dsv-grouped");

        // lines are (groupIdx, winnerIdxInGroup, loserIdxInGroup, [opt] weight)
        // does not support subset loading (because of groupIdx ambiguity)
        // used in fact as a serialization tool for DataProvider's pairs
        TExistsCheckerFactory::TRegistrator<TFSExistsChecker> DsvGroupedWithIdxExistsCheckerReg("dsv-grouped-with-idx");
        TLineDataReaderFactory::TRegistrator<TFileLineDataReader> DsvGroupedWithIdxLineDataReaderReg("dsv-grouped-with-idx");
        TPairsDataLoaderFactory::TRegistrator<TDsvGroupedPairsLoader> DsvGroupedWithIdxPairsDataLoaderReg("dsv-grouped-with-idx");
    }

}

