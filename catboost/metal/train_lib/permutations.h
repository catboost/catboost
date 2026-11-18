#pragma once

#include <catboost/libs/helpers/cpu_random.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/metal/native/metal_multiclass.h>
#include <catboost/metal/native/metal_greedy_trainer.h>
#include <catboost/metal/native/metal_trainer.h>

#include <util/generic/array_ref.h>
#include <util/generic/vector.h>

namespace NCB {
    struct TMetalPermutationState {
        TVector<float> Predictions; // [permutation][object][dimension], flattened
        TVector<float> MvsLambdas;  // [permutation]
        TVector<ui8> MvsValid;      // [permutation]
    };

    // A non-owning adapter for a live scalar or multiclass Metal session. Each permutation
    // keeps its own training cursor and MVS state; the last supplies exported
    // leaf estimates. All matrices share the same feature grid and row order.
    class TMetalDocParallelPermutations {
    public:
        TMetalDocParallelPermutations(
            void* session,
            ui32 rows,
            ui32 features,
            ui64 randomSeed,
            TConstArrayRef<TConstArrayRef<ui8>> binsByPermutation,
            TConstArrayRef<float> initialCursors = {},
            TConstArrayRef<float> mvsLambdas = {},
            TConstArrayRef<ui8> mvsValid = {},
            ui32 approxDimension = 1,
            bool greedy = false)
            : Session(session)
            , Rows(rows)
            , Count(binsByPermutation.size())
            , ApproxDimension(approxDimension)
            , Greedy(greedy)
            , BaseIterationSeed(MakeBaseIterationSeed(randomSeed))
        {
            CB_ENSURE(Session, "A live Metal session is required for permutations");
            CB_ENSURE(Rows > 0 && features > 0, "Metal permutation dimensions must be positive");
            CB_ENSURE(ApproxDimension >= 1 && ApproxDimension <= 64,
                "Metal permutation approximation dimension must be between 1 and 64");
            CB_ENSURE(!Greedy || ApproxDimension == 1, "Metal greedy permutations require scalar cursors");
            CB_ENSURE(binsByPermutation.size() >= 1 && binsByPermutation.size() <= 64,
                "Metal permutation count must be between 1 and 64");
            CB_ENSURE(initialCursors.empty() || initialCursors.size() == ui64(Count) * Rows * ApproxDimension,
                "Metal permutation cursors must contain one prediction per permutation, object, and dimension");
            CB_ENSURE((mvsLambdas.empty() && mvsValid.empty()) ||
                (mvsLambdas.size() == Count && mvsValid.size() == Count),
                "Metal permutation MVS values and flags must both match the permutation count");

            TVector<const ui8*> bins(Count);
            TVector<const float*> cursors;
            if (!initialCursors.empty()) {
                cursors.resize(Count);
            }
            for (ui32 permutation = 0; permutation < Count; ++permutation) {
                CB_ENSURE(binsByPermutation[permutation].size() == ui64(features) * Rows,
                    "Metal permutation matrices must share the original feature and object dimensions");
                bins[permutation] = binsByPermutation[permutation].data();
                if (!initialCursors.empty()) {
                    cursors[permutation] = initialCursors.data() + ui64(permutation) * Rows * ApproxDimension;
                }
            }

            // The runtime copies all inputs. Without saved cursors, it clones
            // the cursor initialized when the session was created.
            char error[2048] = {};
            const auto configure = Greedy ? cbm_greedy_session_set_permutations : ApproxDimension == 1
                ? cbm_session_set_permutations : cbm_multiclass_session_set_permutations;
            CB_ENSURE(configure(Session, Count, bins.data(),
                cursors.empty() ? nullptr : cursors.data(),
                mvsLambdas.empty() ? nullptr : mvsLambdas.data(),
                mvsValid.empty() ? nullptr : mvsValid.data(), error, sizeof(error)) == 0,
                "Metal permutation initialization failed: " << error);
        }

        ui32 GetCount() const {
            return Count;
        }

        ui32 GetEstimationPermutation() const {
            return Count - 1;
        }

        ui32 SelectForIteration(ui64 absoluteIteration) const {
            // CUDA methods/doc_parallel_boosting.h::Fit creates a fresh RNG
            // for each absolute iteration, so restored runs need no RNG state.
            TRandom random(BaseIterationSeed + absoluteIteration);
            random.Advance(10);
            const ui32 learnPermutationCount = Count > 1 ? Count - 1 : 1;
            // Preserve CUDA's literal (learnPermutationCount - 1) modulus.
            // For P=4, search chooses 0 or 1, while estimation/export uses 3.
            const ui32 selected = learnPermutationCount > 1
                ? static_cast<ui32>(random.NextUniformL() % (learnPermutationCount - 1))
                : 0;
            char error[2048] = {};
            const auto select = Greedy ? cbm_greedy_session_select_permutation : ApproxDimension == 1
                ? cbm_session_select_permutation : cbm_multiclass_session_select_permutation;
            CB_ENSURE(select(Session, selected, error, sizeof(error)) == 0,
                "Metal permutation selection failed at iteration " << absoluteIteration << ": " << error);
            return selected;
        }

        TMetalPermutationState CopyState() const {
            TMetalPermutationState result;
            result.Predictions.resize(ui64(Count) * Rows * ApproxDimension);
            result.MvsLambdas.resize(Count);
            result.MvsValid.resize(Count);
            char error[2048] = {};
            const auto copy = Greedy ? cbm_greedy_session_copy_permutation_state : ApproxDimension == 1
                ? cbm_session_copy_permutation_state : cbm_multiclass_session_copy_permutation_state;
            CB_ENSURE(copy(Session, Count, result.Predictions.data(),
                result.MvsLambdas.data(), result.MvsValid.data(), error, sizeof(error)) == 0,
                "Metal permutation state copy failed: " << error);
            return result;
        }

    private:
        static ui64 MakeBaseIterationSeed(ui64 randomSeed) {
            // CUDA train_lib/train.cpp constructs TGpuAwareRandom(random_seed),
            // then TBoosting's constructor takes its first NextUniformL draw.
            TRandom random(randomSeed);
            return random.NextUniformL();
        }

        void* const Session;
        const ui32 Rows;
        const ui32 Count;
        const ui32 ApproxDimension;
        const bool Greedy;
        const ui64 BaseIterationSeed;
    };
}
