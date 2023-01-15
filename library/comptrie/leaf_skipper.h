#pragma once

#include <cstddef>

namespace NCompactTrie {
    class ILeafSkipper {
    public:
        virtual size_t SkipLeaf(const char* p) const = 0;
        virtual ~ILeafSkipper() = default;
    };

    template <class TPacker>
    class TPackerLeafSkipper: public ILeafSkipper {
    private:
        const TPacker* Packer;

    public:
        TPackerLeafSkipper(const TPacker* packer)
            : Packer(packer)
        {
        }

        size_t SkipLeaf(const char* p) const override {
            return Packer->SkipLeaf(p);
        }

        // For test purposes.
        const TPacker* GetPacker() const {
            return Packer;
        }
    };

    // The data you need to traverse the trie without unpacking the values.
    struct TOpaqueTrie {
        const char* Data;
        size_t Length;
        const ILeafSkipper& SkipFunction;

        TOpaqueTrie(const char* data, size_t dataLength, const ILeafSkipper& skipFunction)
            : Data(data)
            , Length(dataLength)
            , SkipFunction(skipFunction)
        {
        }

        bool operator==(const TOpaqueTrie& other) const {
            return Data == other.Data &&
                   Length == other.Length &&
                   &SkipFunction == &other.SkipFunction;
        }

        bool operator!=(const TOpaqueTrie& other) const {
            return !(*this == other);
        }
    };
}
