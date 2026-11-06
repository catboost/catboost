#pragma once

// Read-only view of the static resource table produced at link time by
// library/python/runtime_py3/pyres_ld_plugin.py (see the layout description there).
// Dependency-free on purpose: usable from the registry and from standalone tests.
//
//   header (64 bytes)
//       0  char[8]  magic "YPYRES01"
//       8  u64      count
//      16  u64      nbuckets      power of two, >= 2 * count
//      24  u64      buckets_off   nbuckets x (u64 hash, u64 key_off); key_off == 0 -> empty
//      32  u64      order_off     count x u64 key_off, sorted by key bytes
//      40  u64      keys_off      key records (8-aligned): u32 klen, u32 vlen, u64 val_off, key bytes
//      48  u64      total_size
//      56  u64      values_off    raw values (compressed as in NResource::Compress) at val_off
//
// Keys are kept apart from values so that enumerating keys does not page in the values.
//
// Hash is CityHash64 of the key, i.e. NHashPrivate::ComputeStringHash on 64-bit platforms.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

namespace NResource::NStaticHash {
    constexpr std::string_view MAGIC = "YPYRES01";

    struct TRecord {
        std::string_view Key;
        std::string_view Value;
    };

    class TTable {
    public:
        static const TTable* FromBlob(const void* blob) noexcept {
            if (!blob) {
                return nullptr;
            }
            const auto* table = static_cast<const TTable*>(blob);
            if (memcmp(table->Magic_, MAGIC.data(), MAGIC.size()) != 0) {
                return nullptr;
            }
            return table;
        }

        size_t Count() const noexcept {
            return Count_;
        }

        // Records in key order.
        TRecord ByIndex(size_t idx) const noexcept {
            return RecordAt(Load64(Base() + OrderOff_ + 8 * idx));
        }

        bool FindByHash(uint64_t hash, std::string_view key, TRecord* out) const noexcept {
            if (Count_ == 0) {
                return false;
            }
            const uint64_t mask = NBuckets_ - 1;
            const char* buckets = Base() + BucketsOff_;
            for (uint64_t i = hash & mask;; i = (i + 1) & mask) {
                const char* bucket = buckets + 16 * i;
                const uint64_t off = Load64(bucket + 8);
                if (off == 0) {
                    return false;
                }
                if (Load64(bucket) == hash) {
                    const TRecord rec = RecordAt(off);
                    if (rec.Key == key) {
                        *out = rec;
                        return true;
                    }
                }
            }
        }

        size_t SizeInBytes() const noexcept {
            return TotalSize_;
        }

    private:
        const char* Base() const noexcept {
            return reinterpret_cast<const char*>(this);
        }

        static uint64_t Load64(const char* p) noexcept {
            uint64_t v;
            memcpy(&v, p, sizeof(v));
            return v;
        }

        static uint32_t Load32(const char* p) noexcept {
            uint32_t v;
            memcpy(&v, p, sizeof(v));
            return v;
        }

        TRecord RecordAt(uint64_t off) const noexcept {
            const char* p = Base() + off;
            const uint32_t klen = Load32(p);
            const uint32_t vlen = Load32(p + 4);
            const uint64_t voff = Load64(p + 8);
            return {std::string_view(p + 16, klen), std::string_view(Base() + voff, vlen)};
        }

        TTable() = delete;
        TTable(const TTable&) = delete;
        ~TTable() = delete;

        char Magic_[8];
        uint64_t Count_;
        uint64_t NBuckets_;
        uint64_t BucketsOff_;
        uint64_t OrderOff_;
        [[maybe_unused]] uint64_t KeysOff_;
        uint64_t TotalSize_;
        [[maybe_unused]] uint64_t ValuesOff_;
    };

    static_assert(sizeof(TTable) == 64);
} // namespace NResource::NStaticHash
