/*
 * Copyright 2014 Google Inc. All rights reserved.
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
 */

// independent from idl_parser, since this code is not needed for most clients

#include <unordered_set>

#include "flatbuffers/code_generators.h"
#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/flatc.h"
#include "flatbuffers/idl.h"
#include "flatbuffers/util.h"

namespace flatbuffers {

// Pedantic warning free version of toupper().
inline char ToUpper(char c) {
  return static_cast<char>(::toupper(static_cast<unsigned char>(c)));
}

static std::string GeneratedIterFileName(const std::string &path,
                                     const std::string &file_name) {
  return path + file_name + ".iter.fbs.h";
}

namespace cpp_yandex_maps_iter {
class CppIterGenerator : public BaseGenerator {
 public:
  CppIterGenerator(const Parser &parser, const std::string &path,
               const std::string &file_name)
      : BaseGenerator(parser, path, file_name, "", "::", "h"),
        cur_name_space_(nullptr) {
    static const char *const keywords[] = {
      "alignas",
      "alignof",
      "and",
      "and_eq",
      "asm",
      "atomic_cancel",
      "atomic_commit",
      "atomic_noexcept",
      "auto",
      "bitand",
      "bitor",
      "bool",
      "break",
      "case",
      "catch",
      "char",
      "char16_t",
      "char32_t",
      "class",
      "compl",
      "concept",
      "const",
      "constexpr",
      "const_cast",
      "continue",
      "co_await",
      "co_return",
      "co_yield",
      "decltype",
      "default",
      "delete",
      "do",
      "double",
      "dynamic_cast",
      "else",
      "enum",
      "explicit",
      "export",
      "extern",
      "false",
      "float",
      "for",
      "friend",
      "goto",
      "if",
      "import",
      "inline",
      "int",
      "long",
      "module",
      "mutable",
      "namespace",
      "new",
      "noexcept",
      "not",
      "not_eq",
      "nullptr",
      "operator",
      "or",
      "or_eq",
      "private",
      "protected",
      "public",
      "register",
      "reinterpret_cast",
      "requires",
      "return",
      "short",
      "signed",
      "sizeof",
      "static",
      "static_assert",
      "static_cast",
      "struct",
      "switch",
      "synchronized",
      "template",
      "this",
      "thread_local",
      "throw",
      "true",
      "try",
      "typedef",
      "typeid",
      "typename",
      "union",
      "unsigned",
      "using",
      "virtual",
      "void",
      "volatile",
      "wchar_t",
      "while",
      "xor",
      "xor_eq",
      nullptr,
    };
    for (auto kw = keywords; *kw; kw++) keywords_.insert(*kw);
  }

  std::string GenIncludeGuard() const {
    // Generate include guard.
    std::string guard = file_name_;
    // Remove any non-alpha-numeric characters that may appear in a filename.
    struct IsAlnum {
      bool operator()(char c) const { return !isalnum(c); }
    };
    guard.erase(std::remove_if(guard.begin(), guard.end(), IsAlnum()),
                guard.end());
    guard = "FLATBUFFERS_GENERATED_" + guard;
    guard += "_";
    // For further uniqueness, also add the namespace.
    auto name_space = parser_.current_namespace_;
    for (auto it = name_space->components.begin();
         it != name_space->components.end(); ++it) {
      guard += *it + "_";
    }
    guard += "ITER_";
    guard += "H_";
    std::transform(guard.begin(), guard.end(), guard.begin(), ToUpper);
    return guard;
  }

  void GenIncludeDependencies() {
    int num_includes = 0;
    for (auto it = parser_.native_included_files_.begin();
         it != parser_.native_included_files_.end(); ++it) {
      code_ += "#include \"" + *it + "\"";
      num_includes++;
    }
    for (auto it = parser_.included_files_.begin();
         it != parser_.included_files_.end(); ++it) {
      if (it->second.empty()) continue;
      auto noext = flatbuffers::StripExtension(it->second);
      auto basename = flatbuffers::StripPath(noext);

      code_ += "#include \"" + parser_.opts.include_prefix +
               (parser_.opts.keep_prefix ? noext : basename) +
               ".iter.fbs.h\"";
      num_includes++;
    }
    if (num_includes) code_ += "";
  }

  std::string EscapeKeyword(const std::string &name) const {
    return keywords_.find(name) == keywords_.end() ? name : name + "_";
  }

  std::string Name(const Definition &def) const {
    return EscapeKeyword(def.name);
  }

  std::string Name(const EnumVal &ev) const { return EscapeKeyword(ev.name); }

  // Iterate through all definitions we haven't generate code for (enums,
  // structs, and tables) and output them to a single file.
  bool generate() {
    code_.Clear();
    code_ += "// " + std::string(FlatBuffersGeneratedWarning()) + "\n\n";

    const auto include_guard = GenIncludeGuard();
    code_ += "#ifndef " + include_guard;
    code_ += "#define " + include_guard;
    code_ += "";

    if (parser_.opts.gen_nullable) {
      code_ += "#pragma clang system_header\n\n";
    }

    code_ += "#include \"" + file_name_ + ".fbs.h\"";
    code_ += "#include \"contrib/libs/flatbuffers/include/flatbuffers/flatbuffers_iter.h\"";
    code_ += "";

    if (parser_.opts.include_dependence_headers) { GenIncludeDependencies(); }

    FLATBUFFERS_ASSERT(!cur_name_space_);

    // Generate forward declarations for all structs/tables, since they may
    // have circular references.
    for (auto it = parser_.structs_.vec.begin();
         it != parser_.structs_.vec.end(); ++it) {
      const auto &struct_def = **it;
      if (!struct_def.generated && !struct_def.fixed) {
        SetNameSpace(struct_def.defined_namespace);
        code_ += "template <typename Iter>";
        code_ += "struct " + Name(struct_def) + ";";
        code_ += "";
      }
    }

    for (auto it = parser_.structs_.vec.begin();
         it != parser_.structs_.vec.end(); ++it) {
      const auto &struct_def = **it;
      if (!struct_def.fixed && !struct_def.generated) {
        SetNameSpace(struct_def.defined_namespace);
        GenTable(struct_def);
      }
    }

    // Generate convenient global helper functions:
    if (parser_.root_struct_def_ && !parser_.root_struct_def_->fixed) {
      auto &struct_def = *parser_.root_struct_def_;
      SetNameSpace(struct_def.defined_namespace);
      auto name = Name(struct_def);
      auto qualified_name = cur_name_space_->GetFullyQualifiedName(name);
      auto cpp_name = TranslateNameSpace(qualified_name, true);
      const auto cpp_non_iter_name = TranslateNameSpace(qualified_name);
      const auto cpp_non_iter_getter = TranslateNameSpace(
              parser_.namespaces_.back()->GetFullyQualifiedName("Get"+name));

      code_.SetValue("STRUCT_NAME", name);
      code_.SetValue("CPP_NAME", cpp_name);
      code_.SetValue("CPP_NON_ITER_NAME", cpp_non_iter_name);
      code_.SetValue("CPP_NON_ITER_GETTER", cpp_non_iter_getter);

      // The root datatype accessor:
      code_ += "template <typename Iter>";
      code_ += "inline \\";
      code_ += "std::optional<{{CPP_NAME}}<Iter>> Get{{STRUCT_NAME}}(const Iter& buf) {";
      code_ += "  return yandex::maps::flatbuffers_iter::GetRoot<{{CPP_NAME}}<Iter>, Iter>(buf);";
      code_ += "}";
      code_ += "";

      // The non_iter datatype accessor:
      code_ += "inline \\";
      code_ += "const {{CPP_NON_ITER_NAME}} *Get{{STRUCT_NAME}}(const char *buf) {";
      code_ += "  return {{CPP_NON_ITER_GETTER}}(buf);";
      code_ += "}";
      code_ += "";

      if (parser_.file_identifier_.length()) {
        // Return the identifier
        code_ += "inline const char *{{STRUCT_NAME}}Identifier() {";
        code_ += "  return \"" + parser_.file_identifier_ + "\";";
        code_ += "}";
        code_ += "";

        // Check if a buffer has the identifier.
        code_ += "template <typename Iter>";
        code_ += "inline \\";
        code_ += "bool {{STRUCT_NAME}}BufferHasIdentifier(const Iter& buf) {";
        code_ += "  return yandex::maps::flatbuffers_iter::BufferHasIdentifier(";
        code_ += "      buf, {{STRUCT_NAME}}Identifier());";
        code_ += "}";
        code_ += "";
      }

      // The root verifier.
      if (parser_.file_identifier_.length()) {
        code_.SetValue("ID", name + "Identifier()");
      } else {
        code_.SetValue("ID", "nullptr");
      }

      code_ += "template <typename Iter>";
      code_ += "inline bool Verify{{STRUCT_NAME}}Buffer(";
      code_ += "    yandex::maps::flatbuffers_iter::Verifier<Iter> &verifier) {";
      code_ += "  return verifier.template VerifyBuffer<{{CPP_NAME}}<Iter>>({{ID}});";
      code_ += "}";
      code_ += "";

      if (parser_.file_extension_.length()) {
        // Return the extension
        code_ += "inline const char *{{STRUCT_NAME}}Extension() {";
        code_ += "  return \"" + parser_.file_extension_ + "\";";
        code_ += "}";
        code_ += "";
      }
    }

    if (cur_name_space_) SetNameSpace(nullptr);

    // Close the include guard.
    code_ += "#endif  // " + include_guard;

    const auto file_path = GeneratedIterFileName(path_, file_name_);
    const auto final_code = code_.ToString();
    return SaveFile(file_path.c_str(), final_code, false);
  }

 private:
  CodeWriter code_;

  std::unordered_set<std::string> keywords_;

  // This tracks the current namespace so we can insert namespace declarations.
  const Namespace *cur_name_space_;

  const Namespace *CurrentNameSpace() const { return cur_name_space_; }

// Ensure that a type is prefixed with its namespace whenever it is used
// outside of its namespace.
  std::string WrapInNameSpace(const Namespace *ns,
                                             const std::string &name, bool needIter = false) const {
    if (CurrentNameSpace() == ns) return name;
    std::string qualified_name = qualifying_start_;
    for (auto it = ns->components.begin(); it != ns->components.end(); ++it)
      qualified_name += *it + qualifying_separator_;
    if (needIter)
      qualified_name += "iter" + qualifying_separator_;
    return qualified_name + name;
  }

  std::string WrapInNameSpace(const Definition &def, bool needIter = false) const {
    return WrapInNameSpace(def.defined_namespace, def.name, needIter);
  }

  // Translates a qualified name in flatbuffer text format to the same name in
  // the equivalent C++ namespace.
  static std::string TranslateNameSpace(const std::string &qualified_name, bool needIter = false) {
    std::string cpp_qualified_name = qualified_name;
    size_t start_pos = 0;
    while ((start_pos = cpp_qualified_name.find(".", start_pos)) !=
           std::string::npos) {
      cpp_qualified_name.replace(start_pos, 1, "::");
    }
    if (needIter)
    {
      start_pos = cpp_qualified_name.rfind("::");
      if (start_pos != std::string::npos)
        cpp_qualified_name.replace(start_pos, 2, "::iter::");
    }
    return cpp_qualified_name;
  }

  void GenComment(const std::vector<std::string> &dc, const char *prefix = "") {
    std::string text;
    ::flatbuffers::GenComment(dc, &text, nullptr, prefix);
    code_ += text + "\\";
  }

  // Return a C++ type from the table in idl.h
  std::string GenTypeBasic(const Type &type, bool user_facing_type) const {
    // clang-format off
    static const char * const ctypename[] = {
      #define FLATBUFFERS_TD(ENUM, IDLTYPE, CTYPE, ...) \
        #CTYPE,
        FLATBUFFERS_GEN_TYPES(FLATBUFFERS_TD)
      #undef FLATBUFFERS_TD
    };
    // clang-format on
    if (user_facing_type) {
      if (type.enum_def) return WrapInNameSpace(*type.enum_def);
      if (type.base_type == BASE_TYPE_BOOL) return "bool";
    }
    return ctypename[type.base_type];
  }

  // Return a C++ pointer type, specialized to the actual struct/table types,
  // and vector element types.
  std::string GenTypePointer(const Type &type) const {
    switch (type.base_type) {
      case BASE_TYPE_STRING: {
        return "yandex::maps::flatbuffers_iter::String<Iter>";
      }
      case BASE_TYPE_VECTOR: {
        const auto type_name = GenTypeWire(type.VectorType(), "", false);
        return "yandex::maps::flatbuffers_iter::Vector<" + type_name + ", Iter>";
      }
      case BASE_TYPE_STRUCT: {
        if (IsStruct(type))
          return WrapInNameSpace(*type.struct_def, !type.struct_def->fixed);
        return WrapInNameSpace(*type.struct_def, !type.struct_def->fixed) + "<Iter>";
      }
      case BASE_TYPE_UNION:
      // fall through
      default: { return "void"; }
    }
  }

  // Return a C++ type for any type (scalar/pointer) specifically for
  // building a flatbuffer.
  std::string GenTypeWire(const Type &type, const char *postfix,
                          bool user_facing_type) const {
    if (IsScalar(type.base_type)) {
      return GenTypeBasic(type, user_facing_type) + postfix;
    } else if (IsStruct(type)) {
      return GenTypePointer(type);
    } else {
      return "yandex::maps::flatbuffers_iter::Offset<" + GenTypePointer(type) + ">" + postfix;
    }
  }

  // Return a C++ type for any type (scalar/pointer) that reflects its
  // serialized size.
  std::string GenTypeSize(const Type &type) const {
    if (IsScalar(type.base_type)) {
      return GenTypeBasic(type, false);
    } else if (IsStruct(type)) {
      return GenTypePointer(type);
    } else {
      return "yandex::maps::flatbuffers_iter::uoffset_t";
    }
  }

  // Return a C++ type for any type (scalar/pointer) specifically for
  // using a flatbuffer.
  std::string GenTypeGet(const Type &type, const char *afterbasic,
                         const char *beforeptr, const char *afterptr,
                         bool user_facing_type) {
    if (IsScalar(type.base_type)) {
      return GenTypeBasic(type, user_facing_type) + afterbasic;
    } else {
      return beforeptr + GenTypePointer(type) + afterptr;
    }
  }

  // Generates a value with optionally a cast applied if the field has a
  // different underlying type from its interface type (currently only the
  // case for enums. "from" specify the direction, true meaning from the
  // underlying type to the interface type.
  std::string GenUnderlyingCast(const FieldDef &field, bool from,
                                const std::string &val) {
    if (from && field.value.type.base_type == BASE_TYPE_BOOL) {
      return val + " != 0";
    } else if ((field.value.type.enum_def &&
                IsScalar(field.value.type.base_type)) ||
               field.value.type.base_type == BASE_TYPE_BOOL) {
      return "static_cast<" + GenTypeBasic(field.value.type, from) + ">(" +
             val + ")";
    } else {
      return val;
    }
  }

  std::string GenFieldOffsetName(const FieldDef &field) {
    std::string uname = Name(field);
    std::transform(uname.begin(), uname.end(), uname.begin(), ToUpper);
    return "VT_" + uname;
  }

  std::string GenDefaultConstant(const FieldDef &field) {
    return field.value.type.base_type == BASE_TYPE_FLOAT
               ? field.value.constant + "f"
               : field.value.constant;
  }

  // Generate the code to call the appropriate Verify function(s) for a field.
  void GenVerifyCall(const FieldDef &field, const char *prefix) {
    code_.SetValue("PRE", prefix);
    code_.SetValue("NAME", Name(field));
    code_.SetValue("REQUIRED", field.IsRequired() ? "Required" : "");
    code_.SetValue("SIZE", GenTypeSize(field.value.type));
    code_.SetValue("OFFSET", GenFieldOffsetName(field));
    if (IsScalar(field.value.type.base_type) || IsStruct(field.value.type)) {
      code_.SetValue("ALIGN", NumToString(InlineAlignment(field.value.type)));
      code_ +=
          "{{PRE}}this->template VerifyField{{REQUIRED}}<{{SIZE}}>(verifier, "
          "{{OFFSET}}, {{ALIGN}})\\";
    } else {
      code_.SetValue("OFFSET_SIZE", field.offset64 ? "64" : "");
      code_ +=
          "{{PRE}}this->template VerifyOffset{{REQUIRED}}<{{SIZE}}>(verifier, "
          "{{OFFSET}})\\";
    }

    switch (field.value.type.base_type) {
      case BASE_TYPE_UNION: {
        code_.SetValue("ENUM_NAME", field.value.type.enum_def->name);
        code_.SetValue("SUFFIX", UnionTypeFieldSuffix());
        code_ +=
            "{{PRE}}Verify{{ENUM_NAME}}(verifier, {{NAME}}(), "
            "{{NAME}}{{SUFFIX}}())\\";
        break;
      }
      case BASE_TYPE_STRUCT: {
        if (!field.value.type.struct_def->fixed) {
          code_ += "{{PRE}}verifier.VerifyTable({{NAME}}())\\";
        }
        break;
      }
      case BASE_TYPE_STRING: {
        code_ += "{{PRE}}verifier.Verify({{NAME}}())\\";
        break;
      }
      case BASE_TYPE_VECTOR: {
        code_ += "{{PRE}}verifier.Verify({{NAME}}())\\";

        switch (field.value.type.element) {
          case BASE_TYPE_STRING: {
            code_ += "{{PRE}}verifier.VerifyVectorOfStrings({{NAME}}())\\";
            break;
          }
          case BASE_TYPE_STRUCT: {
            if (!field.value.type.struct_def->fixed) {
              code_ += "{{PRE}}verifier.VerifyVectorOfTables({{NAME}}())\\";
            }
            break;
          }
          case BASE_TYPE_UNION: {
            code_.SetValue("ENUM_NAME", field.value.type.enum_def->name);
            code_ +=
                "{{PRE}}Verify{{ENUM_NAME}}Vector(verifier, {{NAME}}(), "
                "{{NAME}}_type())\\";
            break;
          }
          default: break;
        }
        break;
      }
      default: { break; }
    }
  }

  // Generate CompareWithValue method for a key field.
  void GenKeyFieldMethods(const FieldDef &field) {
    FLATBUFFERS_ASSERT(field.key);
    const bool is_string = (field.value.type.base_type == BASE_TYPE_STRING);

    code_ += "  bool KeyCompareLessThan(const std::optional<{{STRUCT_NAME}}<Iter>>& o) const {";
    if (is_string) {
      // use operator< of flatbuffers::String
      code_ += "    return {{FIELD_NAME}}() < o->{{FIELD_NAME}}();";
    } else {
      code_ += "    return {{FIELD_NAME}}() < o->{{FIELD_NAME}}();";
    }
    code_ += "  }";

    if (is_string) {
      code_ += "  int KeyCompareWithValue(const char *val) const {";
      code_ += "    return strcmp({{FIELD_NAME}}()->str().c_str(), val);";
      code_ += "  }";
    } else {
      FLATBUFFERS_ASSERT(IsScalar(field.value.type.base_type));
      auto type = GenTypeBasic(field.value.type, false);
      if (parser_.opts.scoped_enums && field.value.type.enum_def &&
          IsScalar(field.value.type.base_type)) {
        type = GenTypeGet(field.value.type, " ", "const ", " *", true);
      }
      // Returns {field<val: -1, field==val: 0, field>val: +1}.
      code_.SetValue("KEY_TYPE", type);
      code_ += "  int KeyCompareWithValue({{KEY_TYPE}} val) const {";
      code_ +=
          "    return static_cast<int>({{FIELD_NAME}}() > val) - "
          "static_cast<int>({{FIELD_NAME}}() < val);";
      code_ += "  }";
    }
  }


  // Generate an accessor struct, builder structs & function for a table.
  void GenTable(const StructDef &struct_def) {
    // Generate an accessor struct, with methods of the form:
    // type name() const { return GetField<type>(offset, defaultval); }
    GenComment(struct_def.doc_comment);

    code_.SetValue("STRUCT_NAME", Name(struct_def));
    code_ += "template <typename Iter>";
    code_ +=
        "struct {{STRUCT_NAME}} FLATBUFFERS_FINAL_CLASS"
        " : private yandex::maps::flatbuffers_iter::Table<Iter> {";

    // Generate field id constants.
    if (struct_def.fields.vec.size() > 0) {
      // We need to add a trailing comma to all elements except the last one as
      // older versions of gcc complain about this.
      code_.SetValue("SEP", "");
      code_ += "  enum {";
      for (auto it = struct_def.fields.vec.begin();
           it != struct_def.fields.vec.end(); ++it) {
        const auto &field = **it;
        if (field.deprecated) {
          // Deprecated fields won't be accessible.
          continue;
        }

        code_.SetValue("OFFSET_NAME", GenFieldOffsetName(field));
        code_.SetValue("OFFSET_VALUE", NumToString(field.value.offset));
        code_ += "{{SEP}}    {{OFFSET_NAME}} = {{OFFSET_VALUE}}\\";
        code_.SetValue("SEP", ",\n");
      }
      code_ += "";
      code_ += "  };";
    }

    code_ += "";
    code_ += "  using yandex::maps::flatbuffers_iter::Table<Iter>::Table;";

    // Generate the accessors.
    for (auto it = struct_def.fields.vec.begin();
         it != struct_def.fields.vec.end(); ++it) {
      const auto &field = **it;
      if (field.deprecated) {
        // Deprecated fields won't be accessible.
        continue;
      }

      const bool is_struct = IsStruct(field.value.type);
      const bool is_scalar = IsScalar(field.value.type.base_type);
      code_.SetValue("FIELD_NAME", Name(field));

      // Call a different accessor for pointers, that indirects.
      std::string accessor = "";
      if (is_scalar) {
        accessor = "this->template GetField<";
      } else if (is_struct) {
        accessor = "this->template GetStruct<";
      } else {
        accessor = "this->template GetPointer<";
      }
      auto offset_str = GenFieldOffsetName(field);
      auto offset_type =
          GenTypeGet(field.value.type, "", "", "", false);

      auto call = accessor + offset_type + ">(" + offset_str;
      // Default value as second arg for non-pointer types.
      if (is_scalar) { call += ", " + GenDefaultConstant(field); }
      call += ")";

      GenComment(field.doc_comment, "  ");
      code_.SetValue("FIELD_TYPE",
          GenTypeGet(field.value.type, " ", "std::optional<", "> ", true));
      code_.SetValue("FIELD_VALUE", GenUnderlyingCast(field, true, call));

      code_ += "  {{FIELD_TYPE}}{{FIELD_NAME}}() const {";
      code_ += "    return {{FIELD_VALUE}};";
      code_ += "  }";

      // Generate a comparison function for this field if it is a key.
      if (field.key) {
        GenKeyFieldMethods(field);
      }
    }

    // Generate a verifier function that can check a buffer from an untrusted
    // source will never cause reads outside the buffer.
    code_ += "  bool Verify(yandex::maps::flatbuffers_iter::Verifier<Iter> &verifier) const {";
    code_ += "    return this->VerifyTableStart(verifier)\\";
    for (auto it = struct_def.fields.vec.begin();
         it != struct_def.fields.vec.end(); ++it) {
      const auto &field = **it;
      if (field.deprecated) { continue; }
      GenVerifyCall(field, " &&\n           ");
    }

    code_ += " &&\n           verifier.EndTable();";
    code_ += "  }";

    code_ += "};";  // End of table.
    code_ += "";
  }

  // Set up the correct namespace. Only open a namespace if the existing one is
  // different (closing/opening only what is necessary).
  //
  // The file must start and end with an empty (or null) namespace so that
  // namespaces are properly opened and closed.
  void SetNameSpace(const Namespace *ns) {
    if (cur_name_space_ == ns) { return; }

    // Compute the size of the longest common namespace prefix.
    // If cur_name_space is A::B::C::D and ns is A::B::E::F::G,
    // the common prefix is A::B:: and we have old_size = 4, new_size = 5
    // and common_prefix_size = 2
    size_t old_size = cur_name_space_ ? cur_name_space_->components.size() : 0;
    size_t new_size = ns ? ns->components.size() : 0;

    size_t common_prefix_size = 0;
    while (common_prefix_size < old_size && common_prefix_size < new_size &&
           ns->components[common_prefix_size] ==
               cur_name_space_->components[common_prefix_size]) {
      common_prefix_size++;
    }

    // Close cur_name_space in reverse order to reach the common prefix.
    // In the previous example, D then C are closed.
    if (old_size > 0)
      code_ += "}  // namespace iter";
    for (size_t j = old_size; j > common_prefix_size; --j) {
      code_ += "}  // namespace " + cur_name_space_->components[j - 1];
    }
    if (old_size != common_prefix_size) { code_ += ""; }

    // open namespace parts to reach the ns namespace
    // in the previous example, E, then F, then G are opened
    for (auto j = common_prefix_size; j != new_size; ++j) {
      code_ += "namespace " + ns->components[j] + " {";
    }
    if (new_size > 0)
      code_ += "namespace iter {";
    if (new_size != common_prefix_size) { code_ += ""; }

    cur_name_space_ = ns;
  }
};

}  // namespace cpp_yandex_maps_iter

bool GenerateCPPYandexMapsIter(const Parser &parser, const std::string &path,
                 const std::string &file_name) {
  cpp_yandex_maps_iter::CppIterGenerator generator(parser, path, file_name);
  return generator.generate();
}

namespace cpp_yandex_maps_iter {

class CppIterCodeGenerator : public CodeGenerator {
 public:
  Status GenerateCode(const Parser &parser, const std::string &path,
                      const std::string &filename) override {
    if (!GenerateCPPYandexMapsIter(parser, path, filename)) { return Status::ERROR; }
    return Status::OK;
  }

  Status GenerateCode(
  	const uint8_t* /* buffer */, 
	int64_t /* length */,
	const CodeGenOptions& /* options */
  ) override {
    return Status::NOT_IMPLEMENTED;
  }

  Status GenerateMakeRule(const Parser &parser, const std::string &path,
                          const std::string &filename,
                          std::string &output) override {
    return Status::NOT_IMPLEMENTED;
  }

  Status GenerateGrpcCode(const Parser &parser, const std::string &path,
                          const std::string &filename) override {
    return Status::NOT_IMPLEMENTED;
  }

  Status GenerateRootFile(const Parser &parser,
                          const std::string &path) override {
    (void)parser;
    (void)path;
    return Status::NOT_IMPLEMENTED;
  }

  bool IsSchemaOnly() const override { return true; }

  bool SupportsBfbsGeneration() const override { return false; }

  bool SupportsRootFileGeneration() const override { return false; }

  IDLOptions::Language Language() const override { return IDLOptions::kCppYandexMapsIter; }

  std::string LanguageName() const override { return "C++Iter"; }
};

}  // namespace

std::unique_ptr<CodeGenerator> NewCppYandexMapsIterCodeGenerator() {
  return std::unique_ptr<cpp_yandex_maps_iter::CppIterCodeGenerator>(new cpp_yandex_maps_iter::CppIterCodeGenerator());
}

}  // namespace flatbuffers
