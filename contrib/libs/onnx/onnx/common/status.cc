#include "status.h"
#include <assert.h>
#include "onnx/string_utils.h"

namespace ONNX_NAMESPACE {
namespace Common {

Status::Status(StatusCategory category, int code, const TString& msg) {
  assert(static_cast<int>(StatusCode::OK) != code);
  state_.reset(new State(category, code, msg));
}

Status::Status(StatusCategory category, int code)
    : Status(category, code, EmptyString()) {}

bool Status::IsOK() const noexcept {
  return (state_ == NULL);
}

StatusCategory Status::Category() const noexcept {
  return IsOK() ? StatusCategory::NONE : state_->category;
}

int Status::Code() const noexcept {
  return IsOK() ? static_cast<int>(StatusCode::OK) : state_->code;
}

const TString& Status::ErrorMessage() const {
  return IsOK() ? EmptyString() : state_->msg;
}

TString Status::ToString() const {
  if (state_ == nullptr) {
    return TString("OK");
  }

  TString result;

  if (StatusCategory::CHECKER == state_->category) {
    result += "[CheckerError]";
  } else if (StatusCategory::OPTIMIZER == state_->category) {
    result += "[OptimizerError]";
  }

  result += " : ";
  result += ONNX_NAMESPACE::to_string(Code());
  TString msg;

  switch (static_cast<StatusCode>(Code())) {
    case INVALID_ARGUMENT:
      msg = "INVALID_ARGUMENT";
      break;
    case INVALID_PROTOBUF:
      msg = "INVALID_PROTOBUF";
      break;
    case FAIL:
      msg = "FAIL";
      break;
    default:
      msg = "GENERAL ERROR";
      break;
  }
  result += " : ";
  result += msg;
  result += " : ";
  result += state_->msg;

  return result;
}

const Status& Status::OK() noexcept {
  static Status s_ok;
  return s_ok;
}

const TString& Status::EmptyString() {
  static TString empty_str = "";
  return empty_str;
}

} // namespace Common
} // namespace ONNX_NAMESPACE