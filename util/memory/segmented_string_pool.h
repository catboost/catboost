#pragma once

#include <util/system/align.h>
#include <util/system/yassert.h>
#include <util/system/defaults.h>
#include <util/generic/noncopyable.h>
#include <util/generic/vector.h>
#include <util/generic/strbuf.h>

#include <memory>
#include <cstdio>
#include <cstdlib>

/*
 * Non-reallocated storage for the objects of POD type
 */
template <class T, class Alloc = std::allocator<T>>
class segmented_pool: TNonCopyable {
protected:
    Alloc seg_allocator;
    struct seg_inf {
        T* data;        // allocated chunk
        size_t _size;   // size of allocated chunk in sizeof(T)-units
        size_t freepos; // offset to free chunk's memory in bytes
        seg_inf()
            : data(nullptr)
            , _size(0)
            , freepos(0)
        {
        }
        seg_inf(T* d, size_t sz)
            : data(d)
            , _size(sz)
            , freepos(0)
        {
        }
    };
    using seg_container = TVector<seg_inf>;
    using seg_iterator = typename seg_container::iterator;
    using seg_const_iterator = typename seg_container::const_iterator;
    const size_t segment_size; // default size of a memory chunk in sizeof(T)-units
    size_t last_free;          // size of free memory in chunk in sizeof(T)-units
    size_t last_ins_size;      // size of memory used in chunk by the last append() in bytes
    seg_container segs;        // array of memory chunks
    seg_iterator curseg;       // a segment for the current insertion
    const char* Name;          // for debug memory usage
protected:
    void check_capacity(size_t len) {
        if (Y_UNLIKELY(!last_free || len > last_free)) {
            if (curseg != segs.end() && curseg->freepos > 0) {
                ++curseg;
            }
            last_free = (len > segment_size ? len : segment_size);
            if (curseg == segs.end() || curseg->_size < last_free) {
                segs.push_back(seg_inf(seg_allocator.allocate(last_free), last_free));
                if (Y_UNLIKELY(Name)) {
                    printf("Pool \"%s\" was increased by %" PRISZT " bytes to %" PRISZT " Mb.\n", Name, last_free * sizeof(T), capacity() / 0x100000);
                }
                curseg = segs.end() - 1;
            }
            Y_ASSERT(curseg->freepos == 0);
            Y_ASSERT(curseg->_size >= last_free);
        }
    }

public:
    explicit segmented_pool(size_t segsz, const char* name = nullptr)
        : segment_size(segsz)
        , last_free(0)
        , last_ins_size(0)
        , Name(name)
    {
        curseg = segs.begin();
    }
    ~segmented_pool() {
        clear();
    }
    /* src - array of objects, len - count of elements in array */
    T* append(const T* src, size_t len) {
        check_capacity(len);
        ui8* rv = (ui8*)curseg->data + curseg->freepos;
        last_ins_size = sizeof(T) * len;
        if (src) {
            memcpy(rv, src, last_ins_size);
        }
        curseg->freepos += last_ins_size, last_free -= len;
        return (T*)rv;
    }
    T* append() {
        T* obj = get_raw();
        new (obj) T();
        return obj;
    }
    T* get_raw() { // append(0, 1)
        check_capacity(1);
        ui8* rv = (ui8*)curseg->data + curseg->freepos;
        last_ins_size = sizeof(T);
        curseg->freepos += last_ins_size, last_free -= 1;
        return (T*)rv;
    }
    size_t get_segment_size() const {
        return segment_size;
    }
    bool contains(const T* ptr) const {
        for (seg_const_iterator i = segs.begin(), ie = segs.end(); i != ie; ++i) {
            if ((char*)ptr >= (char*)i->data && (char*)ptr < (char*)i->data + i->freepos) {
                return true;
            }
        }
        return false;
    }
    size_t size() const {
        size_t r = 0;
        for (seg_const_iterator i = segs.begin(); i != segs.end(); ++i) {
            r += i->freepos;
        }
        return r;
    }
    size_t capacity() const {
        return segs.size() * segment_size * sizeof(T);
    }
    void restart() {
        if (curseg != segs.end()) {
            ++curseg;
        }
        for (seg_iterator i = segs.begin(); i != curseg; ++i) {
            i->freepos = 0;
        }
        curseg = segs.begin();
        last_free = 0;
        last_ins_size = 0;
    }
    void clear() {
        for (seg_iterator i = segs.begin(); i != segs.end(); ++i) {
            seg_allocator.deallocate(i->data, i->_size);
        }
        segs.clear();
        curseg = segs.begin();
        last_free = 0;
        last_ins_size = 0;
    }
    void undo_last_append() {
        Y_ASSERT(curseg != segs.end()); // do not use before append()
        if (last_ins_size) {
            Y_ASSERT(last_ins_size <= curseg->freepos);
            curseg->freepos -= last_ins_size;
            last_free += last_ins_size / sizeof(T);
            last_ins_size = 0;
        }
    }
    void alloc_first_seg() {
        Y_ASSERT(capacity() == 0);
        check_capacity(segment_size);
        Y_ASSERT(capacity() == segment_size * sizeof(T));
    }
};

class segmented_string_pool: public segmented_pool<char> {
private:
    using _Base = segmented_pool<char>;

public:
    segmented_string_pool()
        : segmented_string_pool(1024 * 1024)
    {
    }

    explicit segmented_string_pool(size_t segsz)
        : _Base(segsz)
    {
    }
    char* append(const char* src) {
        Y_ASSERT(src);
        return _Base::append(src, strlen(src) + 1);
    }
    char* append(const char* src, size_t len) {
        char* rv = _Base::append(nullptr, len + 1);
        if (src) {
            memcpy(rv, src, len);
        }
        rv[len] = 0;
        return rv;
    }
    char* Append(const TStringBuf s) {
        return append(s.data(), s.size());
    }
    void align_4() {
        size_t t = (curseg->freepos + 3) & ~3;
        last_free -= t - curseg->freepos;
        curseg->freepos = t;
    }
    char* Allocate(size_t len) {
        return append(nullptr, len);
    }
};

template <typename T, typename C>
inline T* pool_push(segmented_pool<C>& pool, const T* v) {
    static_assert(sizeof(C) == 1, "only char type supported");
    size_t len = SizeOf(v);
    C* buf = pool.append(nullptr, AlignUp(len));
    memcpy(buf, v, len);
    return (T*)buf;
}
