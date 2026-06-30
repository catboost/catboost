//
// Copyright 2022 The Abseil Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "y_absl/log/internal/log_sink_set.h"

#ifndef Y_ABSL_HAVE_THREAD_LOCAL
#include <pthread.h>
#endif

#ifdef __ANDROID__
#include <android/log.h>
#endif

#ifdef _WIN32
#include <windows.h>
#endif

#include <algorithm>
#include <vector>

#include "y_absl/base/attributes.h"
#include "y_absl/base/call_once.h"
#include "y_absl/base/config.h"
#include "y_absl/base/internal/raw_logging.h"
#include "y_absl/base/log_severity.h"
#include "y_absl/base/no_destructor.h"
#include "y_absl/base/thread_annotations.h"
#include "y_absl/cleanup/cleanup.h"
#include "y_absl/log/globals.h"
#include "y_absl/log/internal/config.h"
#include "y_absl/log/internal/globals.h"
#include "y_absl/log/log_entry.h"
#include "y_absl/log/log_sink.h"
#include "y_absl/strings/string_view.h"
#include "y_absl/synchronization/mutex.h"
#include "y_absl/types/span.h"

namespace y_absl {
Y_ABSL_NAMESPACE_BEGIN
namespace log_internal {
namespace {

// Returns a mutable reference to a thread-local variable that should be true if
// a globally-registered `LogSink`'s `Send()` is currently being invoked on this
// thread.
bool& ThreadIsLoggingStatus() {
#ifdef Y_ABSL_HAVE_THREAD_LOCAL
  Y_ABSL_CONST_INIT thread_local bool thread_is_logging = false;
  return thread_is_logging;
#else
  Y_ABSL_CONST_INIT static pthread_key_t thread_is_logging_key;
  static const bool unused = [] {
    if (pthread_key_create(&thread_is_logging_key, [](void* data) {
          delete reinterpret_cast<bool*>(data);
        })) {
      perror("pthread_key_create failed!");
      abort();
    }
    return true;
  }();
  (void)unused;  // Fixes -wunused-variable warning
  bool* thread_is_logging_ptr =
      reinterpret_cast<bool*>(pthread_getspecific(thread_is_logging_key));

  if (Y_ABSL_PREDICT_FALSE(!thread_is_logging_ptr)) {
    thread_is_logging_ptr = new bool{false};
    if (pthread_setspecific(thread_is_logging_key, thread_is_logging_ptr)) {
      perror("pthread_setspecific failed");
      abort();
    }
  }
  return *thread_is_logging_ptr;
#endif
}

class StderrLogSink final : public LogSink {
 public:
  ~StderrLogSink() override = default;

  void Send(const y_absl::LogEntry& entry) override {
    if (entry.log_severity() < y_absl::StderrThreshold() &&
        y_absl::log_internal::IsInitialized()) {
      return;
    }

    Y_ABSL_CONST_INIT static y_absl::once_flag warn_if_not_initialized;
    y_absl::call_once(warn_if_not_initialized, []() {
      if (y_absl::log_internal::IsInitialized()) return;
      const char w[] =
          "WARNING: All log messages before y_absl::InitializeLog() is called"
          " are written to STDERR\n";
      y_absl::log_internal::WriteToStderr(w, y_absl::LogSeverity::kWarning);
    });

    if (!entry.stacktrace().empty()) {
      y_absl::log_internal::WriteToStderr(entry.stacktrace(),
                                        entry.log_severity());
    } else {
      // TODO(b/226937039): do this outside else condition once we avoid
      // ReprintFatalMessage
      y_absl::log_internal::WriteToStderr(
          entry.text_message_with_prefix_and_newline(), entry.log_severity());
    }
  }
};

#if defined(__ANDROID__)
class AndroidLogSink final : public LogSink {
 public:
  ~AndroidLogSink() override = default;

  void Send(const y_absl::LogEntry& entry) override {
    const int level = AndroidLogLevel(entry);
    const char* const tag = GetAndroidNativeTag();
    __android_log_write(level, tag,
                        entry.text_message_with_prefix_and_newline_c_str());
    if (entry.log_severity() == y_absl::LogSeverity::kFatal)
      __android_log_write(ANDROID_LOG_FATAL, tag, "terminating.\n");
  }

 private:
  static int AndroidLogLevel(const y_absl::LogEntry& entry) {
    switch (entry.log_severity()) {
      case y_absl::LogSeverity::kFatal:
        return ANDROID_LOG_FATAL;
      case y_absl::LogSeverity::kError:
        return ANDROID_LOG_ERROR;
      case y_absl::LogSeverity::kWarning:
        return ANDROID_LOG_WARN;
      default:
        if (entry.verbosity() >= 2) return ANDROID_LOG_VERBOSE;
        if (entry.verbosity() == 1) return ANDROID_LOG_DEBUG;
        return ANDROID_LOG_INFO;
    }
  }
};
#endif  // !defined(__ANDROID__)

#if defined(_WIN32)
class WindowsDebuggerLogSink final : public LogSink {
 public:
  ~WindowsDebuggerLogSink() override = default;

  void Send(const y_absl::LogEntry& entry) override {
    if (entry.log_severity() < y_absl::StderrThreshold() &&
        y_absl::log_internal::IsInitialized()) {
      return;
    }
    ::OutputDebugStringA(entry.text_message_with_prefix_and_newline_c_str());
  }
};
#endif  // !defined(_WIN32)

class GlobalLogSinkSet final {
 public:
  GlobalLogSinkSet() {
#if defined(__myriad2__) || defined(__Fuchsia__)
    // myriad2 and Fuchsia do not log to stderr by default.
#else
    static y_absl::NoDestructor<StderrLogSink> stderr_log_sink;
    AddLogSink(stderr_log_sink.get());
#endif
#ifdef __ANDROID__
    static y_absl::NoDestructor<AndroidLogSink> android_log_sink;
    AddLogSink(android_log_sink.get());
#endif
#if defined(_WIN32)
    static y_absl::NoDestructor<WindowsDebuggerLogSink> debugger_log_sink;
    AddLogSink(debugger_log_sink.get());
#endif  // !defined(_WIN32)
  }

  void LogToSinks(const y_absl::LogEntry& entry,
                  y_absl::Span<y_absl::LogSink*> extra_sinks, bool extra_sinks_only)
      Y_ABSL_LOCKS_EXCLUDED(guard_) {
    SendToSinks(entry, extra_sinks);

    if (!extra_sinks_only) {
      if (ThreadIsLoggingToLogSink()) {
        y_absl::log_internal::WriteToStderr(
            entry.text_message_with_prefix_and_newline(), entry.log_severity());
      } else {
        y_absl::ReaderMutexLock global_sinks_lock(&guard_);
        ThreadIsLoggingStatus() = true;
        // Ensure the "thread is logging" status is reverted upon leaving the
        // scope even in case of exceptions.
        auto status_cleanup =
            y_absl::MakeCleanup([] { ThreadIsLoggingStatus() = false; });
        SendToSinks(entry, y_absl::MakeSpan(sinks_));
      }
    }
  }

  void AddLogSink(y_absl::LogSink* sink) Y_ABSL_LOCKS_EXCLUDED(guard_) {
    {
      y_absl::WriterMutexLock global_sinks_lock(&guard_);
      auto pos = std::find(sinks_.begin(), sinks_.end(), sink);
      if (pos == sinks_.end()) {
        sinks_.push_back(sink);
        return;
      }
    }
    Y_ABSL_INTERNAL_LOG(FATAL, "Duplicate log sinks are not supported");
  }

  void RemoveLogSink(y_absl::LogSink* sink) Y_ABSL_LOCKS_EXCLUDED(guard_) {
    {
      y_absl::WriterMutexLock global_sinks_lock(&guard_);
      auto pos = std::find(sinks_.begin(), sinks_.end(), sink);
      if (pos != sinks_.end()) {
        sinks_.erase(pos);
        return;
      }
    }
    Y_ABSL_INTERNAL_LOG(FATAL, "Mismatched log sink being removed");
  }

  void FlushLogSinks() Y_ABSL_LOCKS_EXCLUDED(guard_) {
    if (ThreadIsLoggingToLogSink()) {
      // The thread_local condition demonstrates that we're already holding the
      // lock in order to iterate over `sinks_` for dispatch.  The thread-safety
      // annotations don't know this, so we use `Y_ABSL_NO_THREAD_SAFETY_ANALYSIS`
      guard_.AssertReaderHeld();
      FlushLogSinksLocked();
    } else {
      y_absl::ReaderMutexLock global_sinks_lock(&guard_);
      // In case if LogSink::Flush overload decides to log
      ThreadIsLoggingStatus() = true;
      // Ensure the "thread is logging" status is reverted upon leaving the
      // scope even in case of exceptions.
      auto status_cleanup =
          y_absl::MakeCleanup([] { ThreadIsLoggingStatus() = false; });
      FlushLogSinksLocked();
    }
  }

 private:
  void FlushLogSinksLocked() Y_ABSL_SHARED_LOCKS_REQUIRED(guard_) {
    for (y_absl::LogSink* sink : sinks_) {
      sink->Flush();
    }
  }

  // Helper routine for LogToSinks.
  static void SendToSinks(const y_absl::LogEntry& entry,
                          y_absl::Span<y_absl::LogSink*> sinks) {
    for (y_absl::LogSink* sink : sinks) {
      sink->Send(entry);
    }
  }

  using LogSinksSet = std::vector<y_absl::LogSink*>;
  y_absl::Mutex guard_;
  LogSinksSet sinks_ Y_ABSL_GUARDED_BY(guard_);
};

// Returns reference to the global LogSinks set.
GlobalLogSinkSet& GlobalSinks() {
  static y_absl::NoDestructor<GlobalLogSinkSet> global_sinks;
  return *global_sinks;
}

}  // namespace

bool ThreadIsLoggingToLogSink() { return ThreadIsLoggingStatus(); }

void LogToSinks(const y_absl::LogEntry& entry,
                y_absl::Span<y_absl::LogSink*> extra_sinks, bool extra_sinks_only) {
  log_internal::GlobalSinks().LogToSinks(entry, extra_sinks, extra_sinks_only);
}

void AddLogSink(y_absl::LogSink* sink) {
  log_internal::GlobalSinks().AddLogSink(sink);
}

void RemoveLogSink(y_absl::LogSink* sink) {
  log_internal::GlobalSinks().RemoveLogSink(sink);
}

void FlushLogSinks() { log_internal::GlobalSinks().FlushLogSinks(); }

}  // namespace log_internal
Y_ABSL_NAMESPACE_END
}  // namespace y_absl
