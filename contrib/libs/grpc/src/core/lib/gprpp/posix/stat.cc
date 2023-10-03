//
// Copyright 2020 gRPC authors.
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

#include <util/generic/string.h>
#include <util/string/cast.h>

// IWYU pragma: no_include <bits/struct_stat.h>

#include "y_absl/status/status.h"
#include "y_absl/strings/string_view.h"

#ifdef GPR_POSIX_STAT

#include <errno.h>
#include <sys/stat.h>

#include <grpc/support/log.h>

#include "src/core/lib/gprpp/stat.h"
#include "src/core/lib/gprpp/strerror.h"

namespace grpc_core {

y_absl::Status GetFileModificationTime(const char* filename, time_t* timestamp) {
  GPR_ASSERT(filename != nullptr);
  GPR_ASSERT(timestamp != nullptr);
  struct stat buf;
  if (stat(filename, &buf) != 0) {
    TString error_msg = StrError(errno);
    gpr_log(GPR_ERROR, "stat failed for filename %s with error %s.", filename,
            error_msg.c_str());
    return y_absl::Status(y_absl::StatusCode::kInternal, error_msg);
  }
  // Last file/directory modification time.
  *timestamp = buf.st_mtime;
  return y_absl::OkStatus();
}

}  // namespace grpc_core

#endif  // GPR_POSIX_STAT
