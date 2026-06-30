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
//
// -----------------------------------------------------------------------------
// File: log/absl_log.h
// -----------------------------------------------------------------------------
//
// This header declares a family of `Y_ABSL_LOG` macros as alternative spellings
// for macros in `log.h`.
//
// Basic invocation looks like this:
//
//   Y_ABSL_LOG(INFO) << "Found " << num_cookies << " cookies";
//
// Most `Y_ABSL_LOG` macros take a severity level argument. The severity levels
// are `INFO`, `WARNING`, `ERROR`, and `FATAL`.
//
// For full documentation, see comments in `log.h`, which includes full
// reference documentation on use of the equivalent `LOG` macro and has an
// identical set of macros without the Y_ABSL_* prefix.

#ifndef Y_ABSL_LOG_ABSL_LOG_H_
#define Y_ABSL_LOG_ABSL_LOG_H_

#include "y_absl/log/internal/log_impl.h"

#define Y_ABSL_LOG(severity) Y_ABSL_LOG_INTERNAL_LOG_IMPL(_##severity)
#define Y_ABSL_PLOG(severity) Y_ABSL_LOG_INTERNAL_PLOG_IMPL(_##severity)
#define Y_ABSL_DLOG(severity) Y_ABSL_LOG_INTERNAL_DLOG_IMPL(_##severity)

#define Y_ABSL_VLOG(verbose_level) Y_ABSL_LOG_INTERNAL_VLOG_IMPL(verbose_level)
#define Y_ABSL_DVLOG(verbose_level) Y_ABSL_LOG_INTERNAL_DVLOG_IMPL(verbose_level)

#define Y_ABSL_LOG_IF(severity, condition) \
  Y_ABSL_LOG_INTERNAL_LOG_IF_IMPL(_##severity, condition)
#define Y_ABSL_PLOG_IF(severity, condition) \
  Y_ABSL_LOG_INTERNAL_PLOG_IF_IMPL(_##severity, condition)
#define Y_ABSL_DLOG_IF(severity, condition) \
  Y_ABSL_LOG_INTERNAL_DLOG_IF_IMPL(_##severity, condition)

#define Y_ABSL_LOG_EVERY_N(severity, n) \
  Y_ABSL_LOG_INTERNAL_LOG_EVERY_N_IMPL(_##severity, n)
#define Y_ABSL_LOG_FIRST_N(severity, n) \
  Y_ABSL_LOG_INTERNAL_LOG_FIRST_N_IMPL(_##severity, n)
#define Y_ABSL_LOG_EVERY_POW_2(severity) \
  Y_ABSL_LOG_INTERNAL_LOG_EVERY_POW_2_IMPL(_##severity)
#define Y_ABSL_LOG_EVERY_N_SEC(severity, n_seconds) \
  Y_ABSL_LOG_INTERNAL_LOG_EVERY_N_SEC_IMPL(_##severity, n_seconds)

#define Y_ABSL_PLOG_EVERY_N(severity, n) \
  Y_ABSL_LOG_INTERNAL_PLOG_EVERY_N_IMPL(_##severity, n)
#define Y_ABSL_PLOG_FIRST_N(severity, n) \
  Y_ABSL_LOG_INTERNAL_PLOG_FIRST_N_IMPL(_##severity, n)
#define Y_ABSL_PLOG_EVERY_POW_2(severity) \
  Y_ABSL_LOG_INTERNAL_PLOG_EVERY_POW_2_IMPL(_##severity)
#define Y_ABSL_PLOG_EVERY_N_SEC(severity, n_seconds) \
  Y_ABSL_LOG_INTERNAL_PLOG_EVERY_N_SEC_IMPL(_##severity, n_seconds)

#define Y_ABSL_DLOG_EVERY_N(severity, n) \
  Y_ABSL_LOG_INTERNAL_DLOG_EVERY_N_IMPL(_##severity, n)
#define Y_ABSL_DLOG_FIRST_N(severity, n) \
  Y_ABSL_LOG_INTERNAL_DLOG_FIRST_N_IMPL(_##severity, n)
#define Y_ABSL_DLOG_EVERY_POW_2(severity) \
  Y_ABSL_LOG_INTERNAL_DLOG_EVERY_POW_2_IMPL(_##severity)
#define Y_ABSL_DLOG_EVERY_N_SEC(severity, n_seconds) \
  Y_ABSL_LOG_INTERNAL_DLOG_EVERY_N_SEC_IMPL(_##severity, n_seconds)

#define Y_ABSL_VLOG_EVERY_N(verbose_level, n) \
  Y_ABSL_LOG_INTERNAL_VLOG_EVERY_N_IMPL(verbose_level, n)
#define Y_ABSL_VLOG_FIRST_N(verbose_level, n) \
  Y_ABSL_LOG_INTERNAL_VLOG_FIRST_N_IMPL(verbose_level, n)
#define Y_ABSL_VLOG_EVERY_POW_2(verbose_level, n) \
  Y_ABSL_LOG_INTERNAL_VLOG_EVERY_POW_2_IMPL(verbose_level, n)
#define Y_ABSL_VLOG_EVERY_N_SEC(verbose_level, n) \
  Y_ABSL_LOG_INTERNAL_VLOG_EVERY_N_SEC_IMPL(verbose_level, n)

#define Y_ABSL_LOG_IF_EVERY_N(severity, condition, n) \
  Y_ABSL_LOG_INTERNAL_LOG_IF_EVERY_N_IMPL(_##severity, condition, n)
#define Y_ABSL_LOG_IF_FIRST_N(severity, condition, n) \
  Y_ABSL_LOG_INTERNAL_LOG_IF_FIRST_N_IMPL(_##severity, condition, n)
#define Y_ABSL_LOG_IF_EVERY_POW_2(severity, condition) \
  Y_ABSL_LOG_INTERNAL_LOG_IF_EVERY_POW_2_IMPL(_##severity, condition)
#define Y_ABSL_LOG_IF_EVERY_N_SEC(severity, condition, n_seconds) \
  Y_ABSL_LOG_INTERNAL_LOG_IF_EVERY_N_SEC_IMPL(_##severity, condition, n_seconds)

#define Y_ABSL_PLOG_IF_EVERY_N(severity, condition, n) \
  Y_ABSL_LOG_INTERNAL_PLOG_IF_EVERY_N_IMPL(_##severity, condition, n)
#define Y_ABSL_PLOG_IF_FIRST_N(severity, condition, n) \
  Y_ABSL_LOG_INTERNAL_PLOG_IF_FIRST_N_IMPL(_##severity, condition, n)
#define Y_ABSL_PLOG_IF_EVERY_POW_2(severity, condition) \
  Y_ABSL_LOG_INTERNAL_PLOG_IF_EVERY_POW_2_IMPL(_##severity, condition)
#define Y_ABSL_PLOG_IF_EVERY_N_SEC(severity, condition, n_seconds) \
  Y_ABSL_LOG_INTERNAL_PLOG_IF_EVERY_N_SEC_IMPL(_##severity, condition, n_seconds)

#define Y_ABSL_DLOG_IF_EVERY_N(severity, condition, n) \
  Y_ABSL_LOG_INTERNAL_DLOG_IF_EVERY_N_IMPL(_##severity, condition, n)
#define Y_ABSL_DLOG_IF_FIRST_N(severity, condition, n) \
  Y_ABSL_LOG_INTERNAL_DLOG_IF_FIRST_N_IMPL(_##severity, condition, n)
#define Y_ABSL_DLOG_IF_EVERY_POW_2(severity, condition) \
  Y_ABSL_LOG_INTERNAL_DLOG_IF_EVERY_POW_2_IMPL(_##severity, condition)
#define Y_ABSL_DLOG_IF_EVERY_N_SEC(severity, condition, n_seconds) \
  Y_ABSL_LOG_INTERNAL_DLOG_IF_EVERY_N_SEC_IMPL(_##severity, condition, n_seconds)

#endif  // Y_ABSL_LOG_ABSL_LOG_H_
