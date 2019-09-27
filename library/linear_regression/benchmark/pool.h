#pragma once

#include <util/generic/vector.h>
#include <util/generic/string.h>

#include <util/random/mersenne.h>
#include <util/random/shuffle.h>

struct TInstance {
    TVector<double> Features;
    double Goal;
    double Weight;

    static TInstance FromFeaturesString(const TString& featuresString);
};

struct TPool: public TVector<TInstance> {
    enum EIteratorType {
        LearnIterator,
        TestIterator,
    };

    class TCVIterator {
    private:
        const TPool& ParentPool;

        size_t FoldsCount;

        EIteratorType IteratorType;
        size_t TestFoldNumber;

        TVector<size_t> InstanceFoldNumbers;
        const size_t* Current;

        TMersenne<ui64> RandomGenerator;

    public:
        TCVIterator(const TPool& parentPool,
                    const size_t foldsCount,
                    const EIteratorType iteratorType);

        void ResetShuffle();

        void SetTestFold(const size_t testFoldNumber);

        bool IsValid() const;

        const TInstance& operator*() const;
        const TInstance* operator->() const;
        TPool::TCVIterator& operator++();

    private:
        void Advance();
        bool TakeCurrent() const;
    };

    void ReadFromFeatures(const TString& featuresPath);
    TCVIterator CrossValidationIterator(const size_t foldsCount, const EIteratorType iteratorType) const;

    TPool InjurePool(const double injureFactir, const double injureOffset) const;
};
