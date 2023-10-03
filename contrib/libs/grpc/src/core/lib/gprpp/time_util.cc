//
// Copyright 2021 the gRPC authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include <grpc/support/port_platform.h>

#include "src/core/lib/gprpp/time_util.h"

#include <stdint.h>
#include <time.h>

#include <grpc/support/log.h>
#include <grpc/support/time.h>

namespace grpc_core {

gpr_timespec ToGprTimeSpec(y_absl::Duration duration) {
  if (duration == y_absl::InfiniteDuration()) {
    return gpr_inf_future(GPR_TIMESPAN);
  } else if (duration == -y_absl::InfiniteDuration()) {
    return gpr_inf_past(GPR_TIMESPAN);
  } else {
    int64_t s = y_absl::IDivDuration(duration, y_absl::Seconds(1), &duration);
    int64_t n = y_absl::IDivDuration(duration, y_absl::Nanoseconds(1), &duration);
    return gpr_time_add(gpr_time_from_seconds(s, GPR_TIMESPAN),
                        gpr_time_from_nanos(n, GPR_TIMESPAN));
  }
}

gpr_timespec ToGprTimeSpec(y_absl::Time time) {
  if (time == y_absl::InfiniteFuture()) {
    return gpr_inf_future(GPR_CLOCK_REALTIME);
  } else if (time == y_absl::InfinitePast()) {
    return gpr_inf_past(GPR_CLOCK_REALTIME);
  } else {
    timespec ts = y_absl::ToTimespec(time);
    gpr_timespec out;
    out.tv_sec = static_cast<decltype(out.tv_sec)>(ts.tv_sec);
    out.tv_nsec = static_cast<decltype(out.tv_nsec)>(ts.tv_nsec);
    out.clock_type = GPR_CLOCK_REALTIME;
    return out;
  }
}

y_absl::Duration ToAbslDuration(gpr_timespec ts) {
  GPR_ASSERT(ts.clock_type == GPR_TIMESPAN);
  if (gpr_time_cmp(ts, gpr_inf_future(GPR_TIMESPAN)) == 0) {
    return y_absl::InfiniteDuration();
  } else if (gpr_time_cmp(ts, gpr_inf_past(GPR_TIMESPAN)) == 0) {
    return -y_absl::InfiniteDuration();
  } else {
    return y_absl::Seconds(ts.tv_sec) + y_absl::Nanoseconds(ts.tv_nsec);
  }
}

y_absl::Time ToAbslTime(gpr_timespec ts) {
  GPR_ASSERT(ts.clock_type != GPR_TIMESPAN);
  gpr_timespec rts = gpr_convert_clock_type(ts, GPR_CLOCK_REALTIME);
  if (gpr_time_cmp(rts, gpr_inf_future(GPR_CLOCK_REALTIME)) == 0) {
    return y_absl::InfiniteFuture();
  } else if (gpr_time_cmp(rts, gpr_inf_past(GPR_CLOCK_REALTIME)) == 0) {
    return y_absl::InfinitePast();
  } else {
    return y_absl::UnixEpoch() + y_absl::Seconds(rts.tv_sec) +
           y_absl::Nanoseconds(rts.tv_nsec);
  }
}

}  // namespace grpc_core
