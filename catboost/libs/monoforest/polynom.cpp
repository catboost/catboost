#include "polynom.h"

#include <catboost/libs/helpers/set.h>

#include <util/generic/xrange.h>

namespace NMonoForest {
    struct TPathBit {
        int Bits = 0;
        int Sign = 1;
    };

    static inline TVector<TPathBit> LeafToPolynoms(const int path, int maxDepth) {
        TVector<TPathBit> pathBits = {{}};
        for (int depth = 0; depth < maxDepth; ++depth) {
            const int mask = 1 << depth;
            const bool isOne = path & mask;

            if (isOne) {
                for (auto& bit : pathBits) {
                    bit.Bits |= 1 << depth;
                }
            } else {
                ui64 currentPaths = pathBits.size();
                for (ui64 i = 0; i < currentPaths; ++i) {
                    auto bit = pathBits[i];
                    bit.Bits |= 1 << depth;
                    bit.Sign *= -1;
                    pathBits.push_back(bit);
                }
            }
        }
        return pathBits;
    }

    void TPolynomBuilder::AddTree(const TObliviousTree& tree) {
        const auto& treeStructure = tree.GetStructure();

        const int maxDepth = static_cast<int>(treeStructure.GetDepth());
        const int leavesCount = treeStructure.LeavesCount();

        TVector<double> values(leavesCount * tree.OutputDim());

        for (int path = 0; path < leavesCount; ++path) {
            auto polynoms = LeafToPolynoms(path, maxDepth);

            for (const auto& polynom : polynoms) {
                int idx = polynom.Bits;
                for (ui32 dim = 0; dim < tree.OutputDim(); ++dim) {
                    values[idx * tree.OutputDim() + dim] +=
                        polynom.Sign * tree.GetValues()[path * tree.OutputDim() + dim];
                }
            }
        }

        THashMap<TMonomStructure, TMonomStat> monomsEnsemble;

        for (int i = 0; i < leavesCount; ++i) {
            TMonomStructure monomStructure;

            THashMap<int, int> maxBins;
            TVector<TBinarySplit> splits;
            for (int depth = 0; depth < maxDepth; ++depth) {
                int mask = 1 << depth;
                if (i & mask) {
                    auto& split = treeStructure.Splits[depth];
                    maxBins[split.FeatureId] = Max<int>(split.BinIdx, maxBins[split.FeatureId]);
                    splits.push_back(split);
                }
            }
            THashMap<int, int> oheSplits;
            bool degenerate = false;

            for (const auto split : splits) {
                if (split.SplitType == EBinSplitType::TakeBin) {
                    oheSplits[split.FeatureId]++;
                    if (oheSplits[split.FeatureId] > 1) {
                        degenerate = true;
                    }
                    monomStructure.Splits.push_back(split);
                } else {
                    TBinarySplit fixedSplit = split;
                    fixedSplit.BinIdx = maxBins[split.FeatureId];
                    monomStructure.Splits.push_back(fixedSplit);
                }
            }
            if (degenerate) {
                continue;
            }
            SortUnique(monomStructure.Splits);

            int weightMask = 0;
            for (const auto& split : monomStructure.Splits) {
                for (int depth = 0; depth < maxDepth; ++depth) {
                    const auto& treeSplit = treeStructure.Splits[depth];
                    if (treeSplit == split) {
                        weightMask |= 1 << depth;
                    }
                }
            }
            double monomWeight = 0;
            for (int leaf = 0; leaf < leavesCount; ++leaf) {
                if ((leaf & weightMask) == weightMask) {
                    monomWeight += tree.GetWeights()[leaf];
                }
            }

            auto& dst = monomsEnsemble[monomStructure];

            if (dst.Weight < 0) {
                dst.Weight = monomWeight;
            } else {
                CB_ENSURE(dst.Weight == monomWeight,
                          "error: monom weight depends on dataset only: " << monomWeight << " ≠ "
                                                                          << dst.Weight);
            }

            if (dst.Value.size() < tree.OutputDim()) {
                dst.Value.resize(tree.OutputDim());
            }
            for (ui32 dim = 0; dim < tree.OutputDim(); ++dim) {
                dst.Value[dim] += values[i * tree.OutputDim() + dim];
            }
        }

        for (const auto& [structure, stat] : monomsEnsemble) {
            auto& dst = MonomsEnsemble[structure];
            if (dst.Weight < 0) {
                dst.Weight = stat.Weight;
            } else {
                CB_ENSURE(dst.Weight == stat.Weight,
                          "error: monom weight depends on dataset only: " << stat.Weight << " ≠ "
                                                                          << dst.Weight);
            }
            if (dst.Value.size() < tree.OutputDim()) {
                dst.Value.resize(tree.OutputDim());
            }
            for (ui32 k = 0; k < tree.OutputDim(); ++k) {
                dst.Value[k] += stat.Value[k];
            }
        }
    }

    void TPolynomBuilder::AddTree(const TNonSymmetricTree& tree) {
        auto visitor = [&](const TLeafPath& path, TConstArrayRef<float> leaf, double) {
          CB_ENSURE(leaf.size());

          int bits = 0;
          for (ui32 depth = 0; depth < path.GetDepth(); ++depth) {
              if (path.Directions[depth] == ESplitValue::One) {
                  bits |= 1 << depth;
              }
          }
          auto monoms = LeafToPolynoms(bits, path.GetDepth());

          for (const auto& monom : monoms) {
              const int activeFeatures = monom.Bits;
              THashMap<TBinarySplit, ui32> splits;

              for (ui32 i = 0; i < path.GetDepth(); ++i) {
                  if (activeFeatures & (1 << i)) {
                      auto srcSplit = path.Splits[i];

                      if (srcSplit.SplitType == EBinSplitType::TakeGreater) {
                          auto baseSplit = srcSplit;
                          baseSplit.BinIdx = 0;
                          splits[baseSplit] = std::max<ui32>(splits[baseSplit], srcSplit.BinIdx);
                      } else {
                          splits[srcSplit] = srcSplit.BinIdx;
                      }
                  }
              }

              TMonomStructure structure;
              for (const auto& [baseSplit, binIdx] : splits) {
                  auto split = baseSplit;
                  split.BinIdx = binIdx;
                  structure.Splits.push_back(split);
              }
              SortUnique(structure.Splits);
              auto& dst = MonomsEnsemble[structure];
              dst.Value.resize(leaf.size());
              for (ui32 i = 0; i < leaf.size(); ++i) {
                  dst.Value[i] += monom.Sign * leaf[i];
              }
          }
        };
        tree.VisitLeavesAndWeights(visitor);
    }

    TPolynom TPolynomBuilder::Build() {
        return {MonomsEnsemble};
    }

    static inline void AddMonomToTree(const TMonom& monom, const TObliviousTreeStructure& treeStructure, TArrayRef<double> leafValues) {
        const auto outDim = monom.Stat.Value.size();
        const auto& monomSplits = monom.Structure.Splits;
        const auto& treeSplits = treeStructure.Splits;

        TVector<int> bitsToFill;
        int baseLeaf = 0;
        {
            {
                ui32 monomCursor = 0;
                ui32 treeCursor = 0;
                while (treeCursor < treeSplits.size()) {
                    if (monomCursor < monomSplits.size() && monomSplits[monomCursor] == treeSplits[treeCursor]) {
                        baseLeaf |= 1 << treeCursor;
                        ++monomCursor;
                    } else {
                        bitsToFill.push_back(treeCursor);
                    }
                    ++treeCursor;
                }
            }

            const int iterCount = 1 << bitsToFill.size();
            for (int i = 0; i < iterCount; ++i) {
                int leaf = baseLeaf;

                for (ui32 j = 0; j < bitsToFill.size(); ++j) {
                    if (i & (1 << j)) {
                        leaf |= 1 << bitsToFill[j];
                    }
                }
                for (ui32 dim = 0; dim < outDim; ++dim) {
                    leafValues[leaf * outDim + dim] += monom.Stat.Value[dim];
                }
            }
        }
    }

    TAdditiveModel<TObliviousTree> TPolynomToObliviousEnsembleConverter::Convert(const TPolynom& polynom) {
        CATBOOST_DEBUG_LOG << "Polynom size: " << polynom.MonomsEnsemble.size() << Endl;

        TVector<TMonom> monoms;
        monoms.reserve(polynom.MonomsEnsemble.size());
        for (const auto& [structure, stat] : polynom.MonomsEnsemble) {
            monoms.push_back({structure, stat});
        }

        StableSort(monoms.begin(), monoms.end(), [&](const TMonom& left, const TMonom& right) -> bool {
            return left.Structure.GetDepth() > right.Structure.GetDepth();
        });

        TVector<TObliviousTreeStructure> structures;
        TVector<TVector<double>> leaves;
        int outDim = monoms.back().Stat.Value.size();

        for (const auto& monom : monoms) {
            bool addNew = true;
            for (auto i : xrange(structures.size())) {
                if (NCB::IsSubset(monom.Structure.Splits, structures[i].Splits)) {
                    AddMonomToTree(monom, structures[i], leaves[i]);
                    addNew = false;
                    break;
                }
            }
            if (addNew) {
                TObliviousTreeStructure newStructure;
                newStructure.Splits = monom.Structure.Splits;
                if (monom.Structure.GetDepth() == 0) {
                    TBinarySplit fakeSplit;
                    newStructure.Splits = {fakeSplit};
                }
                structures.push_back(newStructure);
                leaves.push_back(TVector<double>(outDim * newStructure.LeavesCount()));
                AddMonomToTree(monom, structures.back(), leaves.back());
            }
        }
        TAdditiveModel<TObliviousTree> result;
        for (auto i : xrange(structures.size())) {
            result.AddWeakModel(TObliviousTree(std::move(structures[i]), std::move(leaves[i]), outDim));
        }
        CATBOOST_DEBUG_LOG << "Generated symmetric tree size: " << result.Size() << Endl;
        return result;
    }
}
