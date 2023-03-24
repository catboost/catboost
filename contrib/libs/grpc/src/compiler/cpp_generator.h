/*
 *
 * Copyright 2015 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#ifndef GRPC_INTERNAL_COMPILER_CPP_GENERATOR_H
#define GRPC_INTERNAL_COMPILER_CPP_GENERATOR_H

// cpp_generator.h/.cc do not directly depend on GRPC/ProtoBuf, such that they
// can be used to generate code for other serialization systems, such as
// FlatBuffers.

#include <memory>
#include <util/generic/string.h>
#include <util/string/cast.h>
#include <vector>

#include "src/compiler/config.h"
#include "src/compiler/schema_interface.h"

#ifdef GRPC_CUSTOM_STRING
#warning GRPC_CUSTOM_STRING is no longer supported. Please use TString.
#endif

namespace grpc {

// Using grpc::string and grpc::to_string is discouraged in favor of
// TString and ::ToString. This is only for legacy code using
// them explictly.
typedef TString string;     // deprecated

}  // namespace grpc

namespace grpc_cpp_generator {

// Contains all the parameters that are parsed from the command line.
struct Parameters {
  // Puts the service into a namespace
  TString services_namespace;
  // Use system includes (<>) or local includes ("")
  bool use_system_headers;
  // Prefix to any grpc include
  TString grpc_search_path;
  // Generate Google Mock code to facilitate unit testing.
  bool generate_mock_code;
  // Google Mock search path, when non-empty, local includes will be used.
  TString gmock_search_path;
  // *EXPERIMENTAL* Additional include files in grpc.pb.h
  std::vector<TString> additional_header_includes;
  // By default, use "pb.h"
  TString message_header_extension;
  // Whether to include headers corresponding to imports in source file.
  bool include_import_headers;
};

// Return the prologue of the generated header file.
TString GetHeaderPrologue(grpc_generator::File* file,
                              const Parameters& params);

// Return the includes needed for generated header file.
TString GetHeaderIncludes(grpc_generator::File* file,
                              const Parameters& params);

// Return the includes needed for generated source file.
TString GetSourceIncludes(grpc_generator::File* file,
                              const Parameters& params);

// Return the epilogue of the generated header file.
TString GetHeaderEpilogue(grpc_generator::File* file,
                              const Parameters& params);

// Return the prologue of the generated source file.
TString GetSourcePrologue(grpc_generator::File* file,
                              const Parameters& params);

// Return the services for generated header file.
TString GetHeaderServices(grpc_generator::File* file,
                              const Parameters& params);

// Return the services for generated source file.
TString GetSourceServices(grpc_generator::File* file,
                              const Parameters& params);

// Return the epilogue of the generated source file.
TString GetSourceEpilogue(grpc_generator::File* file,
                              const Parameters& params);

// Return the prologue of the generated mock file.
TString GetMockPrologue(grpc_generator::File* file,
                            const Parameters& params);

// Return the includes needed for generated mock file.
TString GetMockIncludes(grpc_generator::File* file,
                            const Parameters& params);

// Return the services for generated mock file.
TString GetMockServices(grpc_generator::File* file,
                            const Parameters& params);

// Return the epilogue of generated mock file.
TString GetMockEpilogue(grpc_generator::File* file,
                            const Parameters& params);

// Return the prologue of the generated mock file.
TString GetMockPrologue(grpc_generator::File* file,
                            const Parameters& params);

// Return the includes needed for generated mock file.
TString GetMockIncludes(grpc_generator::File* file,
                            const Parameters& params);

// Return the services for generated mock file.
TString GetMockServices(grpc_generator::File* file,
                            const Parameters& params);

// Return the epilogue of generated mock file.
TString GetMockEpilogue(grpc_generator::File* file,
                            const Parameters& params);

}  // namespace grpc_cpp_generator

#endif  // GRPC_INTERNAL_COMPILER_CPP_GENERATOR_H
