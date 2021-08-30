// Protocol Buffers - Google's data interchange format
// Copyright 2008 Google Inc.  All rights reserved.
// https://developers.google.com/protocol-buffers/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef GOOGLE_PROTOBUF_COMPILER_JAVA_NAME_RESOLVER_H__
#define GOOGLE_PROTOBUF_COMPILER_JAVA_NAME_RESOLVER_H__

#include <map>
#include <string>

#include <google/protobuf/stubs/common.h>

namespace google {
namespace protobuf {
class Descriptor;
class EnumDescriptor;
class FieldDescriptor;
class FileDescriptor;
class ServiceDescriptor;

namespace compiler {
namespace java {

// Indicates how closely the two class names match.
enum NameEquality { NO_MATCH, EXACT_EQUAL, EQUAL_IGNORE_CASE };

// Used to get the Java class related names for a given descriptor. It caches
// the results to avoid redundant calculation across multiple name queries.
// Thread-safety note: This class is *not* thread-safe.
class ClassNameResolver {
 public:
  ClassNameResolver();
  ~ClassNameResolver();

  // Gets the unqualified outer class name for the file.
  TProtoStringType GetFileClassName(const FileDescriptor* file, bool immutable);
  TProtoStringType GetFileClassName(const FileDescriptor* file, bool immutable,
                               bool kotlin);
  // Gets the unqualified immutable outer class name of a file.
  TProtoStringType GetFileImmutableClassName(const FileDescriptor* file);
  // Gets the unqualified default immutable outer class name of a file
  // (converted from the proto file's name).
  TProtoStringType GetFileDefaultImmutableClassName(const FileDescriptor* file);

  // Check whether there is any type defined in the proto file that has
  // the given class name.
  bool HasConflictingClassName(const FileDescriptor* file,
                               const TProtoStringType& classname,
                               NameEquality equality_mode);

  // Gets the name of the outer class that holds descriptor information.
  // Descriptors are shared between immutable messages and mutable messages.
  // Since both of them are generated optionally, the descriptors need to be
  // put in another common place.
  TProtoStringType GetDescriptorClassName(const FileDescriptor* file);

  // Gets the fully-qualified class name corresponding to the given descriptor.
  TProtoStringType GetClassName(const Descriptor* descriptor, bool immutable);
  TProtoStringType GetClassName(const Descriptor* descriptor, bool immutable,
                           bool kotlin);
  TProtoStringType GetClassName(const EnumDescriptor* descriptor, bool immutable);
  TProtoStringType GetClassName(const EnumDescriptor* descriptor, bool immutable,
                           bool kotlin);
  TProtoStringType GetClassName(const ServiceDescriptor* descriptor, bool immutable);
  TProtoStringType GetClassName(const ServiceDescriptor* descriptor, bool immutable,
                           bool kotlin);
  TProtoStringType GetClassName(const FileDescriptor* descriptor, bool immutable);
  TProtoStringType GetClassName(const FileDescriptor* descriptor, bool immutable,
                           bool kotlin);

  template <class DescriptorType>
  TProtoStringType GetImmutableClassName(const DescriptorType* descriptor) {
    return GetClassName(descriptor, true);
  }
  template <class DescriptorType>
  TProtoStringType GetMutableClassName(const DescriptorType* descriptor) {
    return GetClassName(descriptor, false);
  }

  // Gets the fully qualified name of an extension identifier.
  TProtoStringType GetExtensionIdentifierName(const FieldDescriptor* descriptor,
                                         bool immutable);
  TProtoStringType GetExtensionIdentifierName(const FieldDescriptor* descriptor,
                                         bool immutable, bool kotlin);

  // Gets the fully qualified name for generated classes in Java convention.
  // Nested classes will be separated using '$' instead of '.'
  // For example:
  //   com.package.OuterClass$OuterMessage$InnerMessage
  TProtoStringType GetJavaImmutableClassName(const Descriptor* descriptor);
  TProtoStringType GetJavaImmutableClassName(const EnumDescriptor* descriptor);
  TProtoStringType GetKotlinFactoryName(const Descriptor* descriptor);
  TProtoStringType GetKotlinExtensionsClassName(const Descriptor* descriptor);
  TProtoStringType GetJavaMutableClassName(const Descriptor* descriptor);
  TProtoStringType GetJavaMutableClassName(const EnumDescriptor* descriptor);
  // Gets the outer class and the actual class for downgraded mutable messages.
  TProtoStringType GetDowngradedFileClassName(const FileDescriptor* file);
  TProtoStringType GetDowngradedClassName(const Descriptor* descriptor);

 private:
  // Get the full name of a Java class by prepending the Java package name
  // or outer class name.
  TProtoStringType GetClassFullName(const TProtoStringType& name_without_package,
                               const FileDescriptor* file, bool immutable,
                               bool is_own_file);
  TProtoStringType GetClassFullName(const TProtoStringType& name_without_package,
                               const FileDescriptor* file, bool immutable,
                               bool is_own_file, bool kotlin);
  // Get the Java Class style full name of a message.
  TProtoStringType GetJavaClassFullName(const TProtoStringType& name_without_package,
                                   const FileDescriptor* file, bool immutable);
  TProtoStringType GetJavaClassFullName(const TProtoStringType& name_without_package,
                                   const FileDescriptor* file, bool immutable,
                                   bool kotlin);
  // Caches the result to provide better performance.
  std::map<const FileDescriptor*, TProtoStringType>
      file_immutable_outer_class_names_;

  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(ClassNameResolver);
};

}  // namespace java
}  // namespace compiler
}  // namespace protobuf
}  // namespace google

#endif  // GOOGLE_PROTOBUF_COMPILER_JAVA_NAME_RESOLVER_H__
