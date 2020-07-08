#ifndef GOOGLE_PROTOBUF_COMPILER_PERLXS_GENERATOR_H__
#define GOOGLE_PROTOBUF_COMPILER_PERLXS_GENERATOR_H__

#include <iostream>
#include <vector>
#include <map>
#include <set>
#include "descriptor.h"
#include "compiler/code_generator.h"
#include "stubs/common.h"
#include "stubs/strutil.h"

namespace google {
namespace protobuf {

class Descriptor;
class EnumDescriptor;
class EnumValueDescriptor;
class FieldDescriptor;
class ServiceDescriptor;

namespace io { class Printer; }

namespace compiler {

// A couple of the C++ code generator headers are not installed, but
// we need to call into that code in a few places.  We duplicate the
// function prototypes here.

namespace cpp {
  extern string ClassName(const Descriptor* descriptor, bool qualified);
  extern string ClassName(const EnumDescriptor* enum_descriptor, 
			  bool qualified);
  extern string FieldName(const FieldDescriptor* field);
  extern string StripProto(const string& filename);
}

namespace perlxs {

// CodeGenerator implementation for generated Perl/XS protocol buffer
// classes.  If you create your own protocol compiler binary and you
// want it to support Perl/XS output, you can do so by registering an
// instance of this CodeGenerator with the CommandLineInterface in
// your main() function.
class LIBPROTOC_EXPORT PerlXSGenerator : public CodeGenerator {
 public:
  PerlXSGenerator();
  virtual ~PerlXSGenerator();

  // implements CodeGenerator ----------------------------------------
  virtual bool Generate(const FileDescriptor* file,
			const string& parameter,
			OutputDirectory* output_directory,
			string* error) const;
  
  bool ProcessOption(const string& option);

 private:
  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(PerlXSGenerator);

 private:
  void GenerateXS(const FileDescriptor* file,
		  OutputDirectory* output_directory,
		  string& base) const;

  void GenerateMessageXS(const Descriptor* descriptor,
			 OutputDirectory* outdir) const;

  void GenerateMessageModule(const Descriptor* descriptor,
			     OutputDirectory* outdir) const;

  void GenerateMessagePOD(const Descriptor* descriptor,
			  OutputDirectory* outdir) const;

  void GenerateDescriptorClassNamePOD(const Descriptor* descriptor,
				      io::Printer& printer) const;

  void GenerateDescriptorMethodPOD(const Descriptor* descriptor,
				   io::Printer& printer) const;

  void GenerateEnumModule(const EnumDescriptor* enum_descriptor,
			  OutputDirectory* outdir) const;
  
  void GenerateMessageXSFieldAccessors(const FieldDescriptor* field,
				       io::Printer& printer,
				       const string& classname) const;

  void GenerateMessageXSCommonMethods(const Descriptor* descriptor,
				      io::Printer& printer,
				      const string& classname) const;

  void GenerateFileXSTypedefs(const FileDescriptor* file,
			      io::Printer& printer,
			      std::set<const Descriptor*>& seen) const;

  void GenerateMessageXSTypedefs(const Descriptor* descriptor,
				 io::Printer& printer,
				 std::set<const Descriptor*>& seen) const;

  void GenerateMessageStatics(const Descriptor* descriptor,
			      io::Printer& printer) const;
  
  void GenerateMessageXSPackage(const Descriptor* descriptor,
				io::Printer& printer) const;

  void GenerateTypemapInput(const Descriptor* descriptor,
			    io::Printer& printer,
			    const string& svname) const;

  string MessageModuleName(const Descriptor* descriptor) const;

  string MessageClassName(const Descriptor* descriptor) const;

  string EnumClassName(const EnumDescriptor* descriptor) const;

  string PackageName(const string& name, const string& package) const;

  void PerlSVGetHelper(io::Printer& printer,
		       const std::map<string, string>& vars,
		       FieldDescriptor::CppType fieldtype,
		       int depth) const;

  void PODPrintEnumValue(const EnumValueDescriptor *value,
			 io::Printer& printer) const;

  string PODFieldTypeString(const FieldDescriptor* field) const;

  void StartFieldToHashref(const FieldDescriptor * field,
			   io::Printer& printer,
			   std::map<string, string>& vars,
			   int depth) const;

  void FieldToHashrefHelper(io::Printer& printer,
			    std::map<string, string>& vars,
			    const FieldDescriptor* field) const;

  void EndFieldToHashref(const FieldDescriptor * field,
			 io::Printer& printer,
			 std::map<string, string>& vars,
			 int depth) const;

  void MessageToHashref(const Descriptor * descriptor,
			io::Printer& printer,
			std::map<string, string>& vars,
			int depth) const;

  void FieldFromHashrefHelper(io::Printer& printer,
			      std::map<string, string>& vars,
			      const FieldDescriptor * field) const;

  void MessageFromHashref(const Descriptor * descriptor,
			  io::Printer& printer,
			  std::map<string, string>& vars,
			  int depth) const;

 private:
  // --perlxs-package option (if given)
  TProtoStringType perlxs_package_;
};
 
}  // namespace perlxs
}  // namespace compiler
}  // namespace protobuf
}  // namespace google
#endif  // GOOGLE_PROTOBUF_COMPILER_PERLXS_GENERATOR_H__
