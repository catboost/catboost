#ifndef Y_ABSL_DEBUGGING_INTERNAL_STACKTRACE_UNIMPLEMENTED_INL_H_
#define Y_ABSL_DEBUGGING_INTERNAL_STACKTRACE_UNIMPLEMENTED_INL_H_

template <bool IS_STACK_FRAMES, bool IS_WITH_CONTEXT>
static int UnwindImpl(void** /* result */, int* /* sizes */,
                      int /* max_depth */, int /* skip_count */,
                      const void* /* ucp */, int *min_dropped_frames) {
  if (min_dropped_frames != nullptr) {
    *min_dropped_frames = 0;
  }
  return 0;
}

namespace y_absl {
Y_ABSL_NAMESPACE_BEGIN
namespace debugging_internal {
bool StackTraceWorksForTest() {
  return false;
}
}  // namespace debugging_internal
Y_ABSL_NAMESPACE_END
}  // namespace y_absl

#endif  // Y_ABSL_DEBUGGING_INTERNAL_STACKTRACE_UNIMPLEMENTED_INL_H_
