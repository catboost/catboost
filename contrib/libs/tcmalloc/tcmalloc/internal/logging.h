// Copyright 2019 The TCMalloc Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Internal logging and related utility routines.

#ifndef TCMALLOC_INTERNAL_LOGGING_H_
#define TCMALLOC_INTERNAL_LOGGING_H_

#include <stdint.h>
#include <stdlib.h>

#include "absl/base/internal/per_thread_tls.h"
#include "absl/base/optimization.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "tcmalloc/internal/config.h"

//-------------------------------------------------------------------
// Utility routines
//-------------------------------------------------------------------

// Safe logging helper: we write directly to the stderr file
// descriptor and avoid FILE buffering because that may invoke
// malloc().
//
// Example:
//   Log(kLog, __FILE__, __LINE__, "error", bytes);

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {

static constexpr int kMaxStackDepth = 64;

// size/depth are made the same size as a pointer so that some generic
// code below can conveniently cast them back and forth to void*.
struct StackTrace {

  // For small sampled objects, we allocate a full span to hold the
  // sampled object.  However to avoid disturbing fragmentation
  // profiles, in such cases we also allocate a small proxy object
  // using the normal mechanism.
  //
  // proxy field is defined only for heap sample stack traces.
  // For heap samples, proxy==NULL iff size > kMaxSize.
  void* proxy;

  uintptr_t requested_size;
  uintptr_t requested_alignment;
  uintptr_t allocated_size;  // size after sizeclass/page rounding

  uintptr_t depth;           // Number of PC values stored in array below
  void* stack[kMaxStackDepth];

  // weight is the expected number of *bytes* that were requested
  // between the previous sample and this one
  size_t weight;

  void* user_data;

  template <typename H>
  friend H AbslHashValue(H h, const StackTrace& t) {
    // As we use StackTrace as a key-value node in StackTraceTable, we only
    // produce a hasher for the fields used as keys.
    return H::combine(H::combine_contiguous(std::move(h), t.stack, t.depth),
                      t.depth, t.requested_size, t.requested_alignment,
                      t.allocated_size
    );
  }
};

enum LogMode {
  kLog,           // Just print the message
  kLogWithStack,  // Print the message and a stack trace
};

class Logger;

// A LogItem holds any of the argument types that can be passed to Log()
class LogItem {
 public:
  LogItem() : tag_(kEnd) {}
  LogItem(const char* v) : tag_(kStr) { u_.str = v; }
  LogItem(int v) : tag_(kSigned) { u_.snum = v; }
  LogItem(long v) : tag_(kSigned) { u_.snum = v; }
  LogItem(long long v) : tag_(kSigned) { u_.snum = v; }
  LogItem(unsigned int v) : tag_(kUnsigned) { u_.unum = v; }
  LogItem(unsigned long v) : tag_(kUnsigned) { u_.unum = v; }
  LogItem(unsigned long long v) : tag_(kUnsigned) { u_.unum = v; }
  LogItem(const void* v) : tag_(kPtr) { u_.ptr = v; }

 private:
  friend class Logger;
  enum Tag { kStr, kSigned, kUnsigned, kPtr, kEnd };
  Tag tag_;
  union {
    const char* str;
    const void* ptr;
    int64_t snum;
    uint64_t unum;
  } u_;
};

extern void Log(LogMode mode, const char* filename, int line, LogItem a,
                LogItem b = LogItem(), LogItem c = LogItem(),
                LogItem d = LogItem());

enum CrashMode {
  kCrash,          // Print the message and crash
  kCrashWithStats  // Print the message, some stats, and crash
};

ABSL_ATTRIBUTE_NORETURN
void Crash(CrashMode mode, const char* filename, int line, LogItem a,
           LogItem b = LogItem(), LogItem c = LogItem(), LogItem d = LogItem());

// Tests can override this function to collect logging messages.
extern void (*log_message_writer)(const char* msg, int length);

// Like assert(), but executed even in NDEBUG mode
#undef CHECK_CONDITION
#define CHECK_CONDITION(cond)                                           \
  (ABSL_PREDICT_TRUE(cond) ? (void)0                                    \
                           : (::tcmalloc::tcmalloc_internal::Crash(     \
                                 ::tcmalloc::tcmalloc_internal::kCrash, \
                                 __FILE__, __LINE__, #cond)))

// Our own version of assert() so we can avoid hanging by trying to do
// all kinds of goofy printing while holding the malloc lock.
#ifndef NDEBUG
#define ASSERT(cond) CHECK_CONDITION(cond)
#else
#define ASSERT(cond) ((void)0)
#endif

// Print into buffer
class Printer {
 private:
  char* buf_;     // Where should we write next
  int left_;      // Space left in buffer (including space for \0)
  int required_;  // Space we needed to complete all printf calls up to this
                  // point

 public:
  // REQUIRES: "length > 0"
  Printer(char* buf, int length) : buf_(buf), left_(length), required_(0) {
    ASSERT(length > 0);
    buf[0] = '\0';
  }

  template <typename... Args>
  void printf(const absl::FormatSpec<Args...>& format, const Args&... args) {
    ASSERT(left_ >= 0);
    if (left_ <= 0) {
      return;
    }

    const int r = absl::SNPrintF(buf_, left_, format, args...);
    if (r < 0) {
      left_ = 0;
      return;
    }
    required_ += r;

    if (r > left_) {
      left_ = 0;
    } else {
      left_ -= r;
      buf_ += r;
    }
  }

  int SpaceRequired() const { return required_; }
};

enum PbtxtRegionType { kTop, kNested };

// A helper class that prints pbtxt via RAII. A pbtxt region can be either a
// top region (with no brackets) or a nested region (enclosed by curly
// brackets).
class PbtxtRegion {
 public:
  PbtxtRegion(Printer* out, PbtxtRegionType type, int indent);
  ~PbtxtRegion();

  PbtxtRegion(const PbtxtRegion&) = delete;
  PbtxtRegion(PbtxtRegion&&) = default;

  // Prints 'key: value'.
  void PrintI64(absl::string_view key, int64_t value);
  void PrintDouble(absl::string_view key, double value);
  void PrintBool(absl::string_view key, bool value);
  // Useful for enums.
  void PrintRaw(absl::string_view key, absl::string_view value);

  // Prints 'key subregion'. Return the created subregion.
  PbtxtRegion CreateSubRegion(absl::string_view key);

 private:
  void NewLineAndIndent();

  Printer* out_;
  PbtxtRegionType type_;
  int indent_;
};

}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END

#endif  // TCMALLOC_INTERNAL_LOGGING_H_
