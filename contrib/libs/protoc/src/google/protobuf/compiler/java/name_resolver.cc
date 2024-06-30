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

#include <google/protobuf/compiler/java/name_resolver.h>

#include <map>
#include <string>

#include <google/protobuf/compiler/code_generator.h>
#include <google/protobuf/stubs/substitute.h>
#include <google/protobuf/compiler/java/helpers.h>
#include <google/protobuf/compiler/java/names.h>

// Must be last.
#include <google/protobuf/port_def.inc>

namespace google {
namespace protobuf {
namespace compiler {
namespace java {

namespace {
// A suffix that will be appended to the file's outer class name if the name
// conflicts with some other types defined in the file.
const char* kOuterClassNameSuffix = "OuterClass";

// Strip package name from a descriptor's full name.
// For example:
//   Full name   : foo.Bar.Baz
//   Package name: foo
//   After strip : Bar.Baz
TProtoStringType StripPackageName(const TProtoStringType& full_name,
                             const FileDescriptor* file) {
  if (file->package().empty()) {
    return full_name;
  } else {
    // Strip package name
    return full_name.substr(file->package().size() + 1);
  }
}

// Get the name of a message's Java class without package name prefix.
TProtoStringType ClassNameWithoutPackage(const Descriptor* descriptor,
                                    bool immutable) {
  return StripPackageName(descriptor->full_name(), descriptor->file());
}

TProtoStringType ClassNameWithoutPackageKotlin(const Descriptor* descriptor) {
  TProtoStringType result = descriptor->name();
  const Descriptor* temp = descriptor->containing_type();

  while (temp) {
    result = temp->name() + "Kt." + result;
    temp = temp->containing_type();
  }
  return result;
}

// Get the name of an enum's Java class without package name prefix.
TProtoStringType ClassNameWithoutPackage(const EnumDescriptor* descriptor,
                                    bool immutable) {
  // Doesn't append "Mutable" for enum type's name.
  const Descriptor* message_descriptor = descriptor->containing_type();
  if (message_descriptor == NULL) {
    return descriptor->name();
  } else {
    return ClassNameWithoutPackage(message_descriptor, immutable) + "." +
           descriptor->name();
  }
}

// Get the name of a service's Java class without package name prefix.
TProtoStringType ClassNameWithoutPackage(const ServiceDescriptor* descriptor,
                                    bool immutable) {
  TProtoStringType full_name =
      StripPackageName(descriptor->full_name(), descriptor->file());
  // We don't allow nested service definitions.
  GOOGLE_CHECK(full_name.find('.') == TProtoStringType::npos);
  return full_name;
}

// Return true if a and b are equals (case insensitive).
NameEquality CheckNameEquality(const TProtoStringType& a, const TProtoStringType& b) {
  if (ToUpper(a) == ToUpper(b)) {
    if (a == b) {
      return NameEquality::EXACT_EQUAL;
    }
    return NameEquality::EQUAL_IGNORE_CASE;
  }
  return NameEquality::NO_MATCH;
}

// Check whether a given message or its nested types has the given class name.
bool MessageHasConflictingClassName(const Descriptor* message,
                                    const TProtoStringType& classname,
                                    NameEquality equality_mode) {
  if (CheckNameEquality(message->name(), classname) == equality_mode) {
    return true;
  }
  for (int i = 0; i < message->nested_type_count(); ++i) {
    if (MessageHasConflictingClassName(message->nested_type(i), classname,
                                       equality_mode)) {
      return true;
    }
  }
  for (int i = 0; i < message->enum_type_count(); ++i) {
    if (CheckNameEquality(message->enum_type(i)->name(), classname) ==
        equality_mode) {
      return true;
    }
  }
  return false;
}

}  // namespace

ClassNameResolver::ClassNameResolver() {}

ClassNameResolver::~ClassNameResolver() {}

TProtoStringType ClassNameResolver::GetFileDefaultImmutableClassName(
    const FileDescriptor* file) {
  TProtoStringType basename;
  TProtoStringType::size_type last_slash = file->name().find_last_of('/');
  if (last_slash == TProtoStringType::npos) {
    basename = file->name();
  } else {
    basename = file->name().substr(last_slash + 1);
  }
  return UnderscoresToCamelCase(StripProto(basename), true);
}

TProtoStringType ClassNameResolver::GetFileImmutableClassName(
    const FileDescriptor* file) {
  TProtoStringType& class_name = file_immutable_outer_class_names_[file];
  if (class_name.empty()) {
    if (file->options().has_java_outer_classname()) {
      class_name = file->options().java_outer_classname();
    } else {
      class_name = GetFileDefaultImmutableClassName(file);
      if (HasConflictingClassName(file, class_name,
                                  NameEquality::EXACT_EQUAL)) {
        class_name += kOuterClassNameSuffix;
      }
    }
  }
  return class_name;
}

TProtoStringType ClassNameResolver::GetFileClassName(const FileDescriptor* file,
                                                bool immutable) {
  return GetFileClassName(file, immutable, false);
}

TProtoStringType ClassNameResolver::GetFileClassName(const FileDescriptor* file,
                                                bool immutable, bool kotlin) {
  if (kotlin) {
    return GetFileImmutableClassName(file) + "Kt";
  } else if (immutable) {
    return GetFileImmutableClassName(file);
  } else {
    return "Mutable" + GetFileImmutableClassName(file);
  }
}

// Check whether there is any type defined in the proto file that has
// the given class name.
bool ClassNameResolver::HasConflictingClassName(const FileDescriptor* file,
                                                const TProtoStringType& classname,
                                                NameEquality equality_mode) {
  for (int i = 0; i < file->enum_type_count(); i++) {
    if (CheckNameEquality(file->enum_type(i)->name(), classname) ==
        equality_mode) {
      return true;
    }
  }
  for (int i = 0; i < file->service_count(); i++) {
    if (CheckNameEquality(file->service(i)->name(), classname) ==
        equality_mode) {
      return true;
    }
  }
  for (int i = 0; i < file->message_type_count(); i++) {
    if (MessageHasConflictingClassName(file->message_type(i), classname,
                                       equality_mode)) {
      return true;
    }
  }
  return false;
}

TProtoStringType ClassNameResolver::GetDescriptorClassName(
    const FileDescriptor* descriptor) {
  return GetFileImmutableClassName(descriptor);
}

TProtoStringType ClassNameResolver::GetClassName(const FileDescriptor* descriptor,
                                            bool immutable) {
  return GetClassName(descriptor, immutable, false);
}

TProtoStringType ClassNameResolver::GetClassName(const FileDescriptor* descriptor,
                                            bool immutable, bool kotlin) {
  TProtoStringType result = FileJavaPackage(descriptor, immutable);
  if (!result.empty()) result += '.';
  result += GetFileClassName(descriptor, immutable, kotlin);
  return result;
}

// Get the full name of a Java class by prepending the Java package name
// or outer class name.
TProtoStringType ClassNameResolver::GetClassFullName(
    const TProtoStringType& name_without_package, const FileDescriptor* file,
    bool immutable, bool is_own_file) {
  return GetClassFullName(name_without_package, file, immutable, is_own_file,
                          false);
}

TProtoStringType ClassNameResolver::GetClassFullName(
    const TProtoStringType& name_without_package, const FileDescriptor* file,
    bool immutable, bool is_own_file, bool kotlin) {
  TProtoStringType result;
  if (is_own_file) {
    result = FileJavaPackage(file, immutable);
  } else {
    result = GetClassName(file, immutable, kotlin);
  }
  if (!result.empty()) {
    result += '.';
  }
  result += name_without_package;
  if (kotlin) result += "Kt";
  return result;
}

TProtoStringType ClassNameResolver::GetClassName(const Descriptor* descriptor,
                                            bool immutable) {
  return GetClassName(descriptor, immutable, false);
}

TProtoStringType ClassNameResolver::GetClassName(const Descriptor* descriptor,
                                            bool immutable, bool kotlin) {
  return GetClassFullName(
      ClassNameWithoutPackage(descriptor, immutable), descriptor->file(),
      immutable, MultipleJavaFiles(descriptor->file(), immutable), kotlin);
}

TProtoStringType ClassNameResolver::GetClassName(const EnumDescriptor* descriptor,
                                            bool immutable) {
  return GetClassName(descriptor, immutable, false);
}

TProtoStringType ClassNameResolver::GetClassName(const EnumDescriptor* descriptor,
                                            bool immutable, bool kotlin) {
  return GetClassFullName(
      ClassNameWithoutPackage(descriptor, immutable), descriptor->file(),
      immutable, MultipleJavaFiles(descriptor->file(), immutable), kotlin);
}

TProtoStringType ClassNameResolver::GetClassName(const ServiceDescriptor* descriptor,
                                            bool immutable) {
  return GetClassName(descriptor, immutable, false);
}

TProtoStringType ClassNameResolver::GetClassName(const ServiceDescriptor* descriptor,
                                            bool immutable, bool kotlin) {
  return GetClassFullName(ClassNameWithoutPackage(descriptor, immutable),
                          descriptor->file(), immutable,
                          IsOwnFile(descriptor, immutable), kotlin);
}

// Get the Java Class style full name of a message.
TProtoStringType ClassNameResolver::GetJavaClassFullName(
    const TProtoStringType& name_without_package, const FileDescriptor* file,
    bool immutable) {
  return GetJavaClassFullName(name_without_package, file, immutable, false);
}

TProtoStringType ClassNameResolver::GetJavaClassFullName(
    const TProtoStringType& name_without_package, const FileDescriptor* file,
    bool immutable, bool kotlin) {
  TProtoStringType result;
  if (MultipleJavaFiles(file, immutable)) {
    result = FileJavaPackage(file, immutable);
    if (!result.empty()) result += '.';
  } else {
    result = GetClassName(file, immutable, kotlin);
    if (!result.empty()) result += '$';
  }
  result += StringReplace(name_without_package, ".", "$", true);
  return result;
}

TProtoStringType ClassNameResolver::GetExtensionIdentifierName(
    const FieldDescriptor* descriptor, bool immutable) {
  return GetExtensionIdentifierName(descriptor, immutable, false);
}

TProtoStringType ClassNameResolver::GetExtensionIdentifierName(
    const FieldDescriptor* descriptor, bool immutable, bool kotlin) {
  return GetClassName(descriptor->containing_type(), immutable, kotlin) + "." +
         descriptor->name();
}

TProtoStringType ClassNameResolver::GetKotlinFactoryName(
    const Descriptor* descriptor) {
  TProtoStringType name = ToCamelCase(descriptor->name(), /* lower_first = */ true);
  return IsForbiddenKotlin(name) ? name + "_" : name;
}

TProtoStringType ClassNameResolver::GetJavaImmutableClassName(
    const Descriptor* descriptor) {
  return GetJavaClassFullName(ClassNameWithoutPackage(descriptor, true),
                              descriptor->file(), true);
}

TProtoStringType ClassNameResolver::GetJavaImmutableClassName(
    const EnumDescriptor* descriptor) {
  return GetJavaClassFullName(ClassNameWithoutPackage(descriptor, true),
                              descriptor->file(), true);
}

TProtoStringType ClassNameResolver::GetKotlinExtensionsClassName(
    const Descriptor* descriptor) {
  return GetClassFullName(ClassNameWithoutPackageKotlin(descriptor),
                          descriptor->file(), true, true, true);
}

TProtoStringType ClassNameResolver::GetJavaMutableClassName(
    const Descriptor* descriptor) {
  return GetJavaClassFullName(ClassNameWithoutPackage(descriptor, false),
                              descriptor->file(), false);
}

TProtoStringType ClassNameResolver::GetJavaMutableClassName(
    const EnumDescriptor* descriptor) {
  return GetJavaClassFullName(ClassNameWithoutPackage(descriptor, false),
                              descriptor->file(), false);
}

TProtoStringType ClassNameResolver::GetDowngradedFileClassName(
    const FileDescriptor* file) {
  return "Downgraded" + GetFileClassName(file, false);
}

TProtoStringType ClassNameResolver::GetDowngradedClassName(
    const Descriptor* descriptor) {
  return FileJavaPackage(descriptor->file()) + "." +
         GetDowngradedFileClassName(descriptor->file()) + "." +
         ClassNameWithoutPackage(descriptor, false);
}

}  // namespace java
}  // namespace compiler
}  // namespace protobuf
}  // namespace google

#include <google/protobuf/port_undef.inc>
