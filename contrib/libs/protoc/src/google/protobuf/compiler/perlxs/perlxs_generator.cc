#include <iostream>
#include <sstream>
#include <memory>

#include <google/protobuf/compiler/perlxs/perlxs_generator.h>
#include <google/protobuf/compiler/perlxs/perlxs_helpers.h>
#include <google/protobuf/descriptor.h>
#include "google/protobuf/descriptor.pb.h"
#include <google/protobuf/stubs/common.h>
#include <google/protobuf/io/printer.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/io/zero_copy_stream.h>

namespace google {
namespace protobuf {
namespace compiler {
namespace perlxs {

inline bool HasFieldPresence(const FileDescriptor* file) {
  return file->syntax() != FileDescriptor::SYNTAX_PROTO3;
}

bool HasHasMethod(const FieldDescriptor* field) {
  if (HasFieldPresence(field->file())) {
    // In proto1/proto2, every field has a has_$name$() method.
    return true;
  }
  // For message types without true field presence, only fields with a message
  // type have a has_$name$() method.
  return field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE;
}


PerlXSGenerator::PerlXSGenerator() {}
PerlXSGenerator::~PerlXSGenerator() {}


bool
PerlXSGenerator::Generate(const FileDescriptor* file,
			  const TProtoStringType& parameter,
			  OutputDirectory* outdir,
			  TProtoStringType* error) const
{
  // Each top-level message get its own XS source file, Perl module,
  // and typemap.  Each top-level enum gets its own Perl module.  The
  // files are generated in the perlxs_out directory.

  for ( int i = 0; i < file->message_type_count(); i++ ) {
    const Descriptor* message_type = file->message_type(i);

    GenerateMessageXS(message_type, outdir);
    GenerateMessagePOD(message_type, outdir);
    GenerateMessageModule(message_type, outdir);
  }

  for ( int i = 0; i < file->enum_type_count(); i++ ) {
    const EnumDescriptor* enum_type = file->enum_type(i);

    GenerateEnumModule(enum_type, outdir);
  }

  return true;
}

bool
PerlXSGenerator::ProcessOption(const TProtoStringType& option)
{
  size_t equals;
  bool   recognized = false;

  equals = option.find_first_of('=');
  if (equals != TProtoStringType::npos) {
    TProtoStringType name = option.substr(0, equals);
    TProtoStringType value;

    if (option.length() > equals) {
      value = option.substr(equals + 1);
    }

    // Right now, we only recognize the --perlxs-package option.
    if (name == "--perlxs-package") {
      perlxs_package_ = value;
      recognized = true;
    }
  }

  return recognized;
}

void
PerlXSGenerator::GenerateMessageXS(const Descriptor* descriptor,
				   OutputDirectory* outdir) const
{
  TProtoStringType filename = descriptor->name() + ".xs";
  std::unique_ptr<io::ZeroCopyOutputStream> output(outdir->Open(filename));
  io::Printer printer(output.get(), '$'); // '$' works well in the .xs file

  TProtoStringType base = cpp::StripProto(descriptor->file()->name());

  // Boilerplate at the top of the file.

  printer.Print(
		"#include <stdint.h>\n"
		"#include <sstream>\n"
		"#include <google/protobuf/stubs/common.h>\n"
		"#include <google/protobuf/io/zero_copy_stream.h>\n"
		"#include \"$base$.pb.h\"\n"
        "#ifdef __cplusplus\n"
		"extern \"C\" {\n"
		"#endif\n"
		"#include \"EXTERN.h\"\n"
		"#include \"perl.h\"\n"
		"#include \"XSUB.h\"\n"
		"#ifdef __cplusplus\n"
		"}\n"
		"#endif\n"
		"#ifdef do_open\n"
		"#undef do_open\n"
		"#endif\n"
		"#ifdef do_close\n"
		"#undef do_close\n"
		"#endif\n"
		"#ifdef New\n"
		"#undef New\n"
		"#endif\n"
		"\n"
		// "using namespace std;\n"
		"\n",
		"base",
		base);

  // ZeroCopyOutputStream implementation (for improved pack() performance)

  printer.Print("class $base$_OutputStream :\n"
		"  public google::protobuf::io::ZeroCopyOutputStream {\n"
		"public:\n"
		"  explicit $base$_OutputStream(SV * sv) :\n"
		"  sv_(sv), len_(0) {}\n"
		"  ~$base$_OutputStream() {}\n"
		"\n"
		"  bool Next(void** data, int* size)\n"
		"  {\n"
		"    STRLEN nlen = len_ << 1;\n"
		"\n"
		"    if ( nlen < 16 ) nlen = 16;\n"
		"    SvGROW(sv_, nlen);\n"
		"    *data = SvEND(sv_) + len_;\n"
		"    *size = SvLEN(sv_) - len_;\n"
		"    len_ = nlen;\n"
		"\n"
		"    return true;\n"
		"  }\n"
		"\n"
		"  void BackUp(int count)\n"
		"  {\n"
		"    SvCUR_set(sv_, SvLEN(sv_) - count);\n"
		"    len_ -= count;\n"
		"  }\n"
		"\n"
		"  void Sync() {\n"
		"    if ( SvCUR(sv_) == 0 ) {\n"
		"      SvCUR_set(sv_, len_);\n"
		"    }\n"
		"  }\n"
		"\n"
		"  int64_t ByteCount() const\n"
		"  {\n"
		"    return (arc_i64)SvCUR(sv_);\n"
		"  }\n"
		"\n"
		"private:\n"
		"  SV * sv_;\n"
		"  STRLEN len_;\n"
		"\n"
		"  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS($base$_OutputStream);\n"
		"};\n"
		"\n"
		"\n",
		"base",
		base);

  // Typedefs, Statics, and XS packages

  std::set<const Descriptor*> seen;

  GenerateFileXSTypedefs(descriptor->file(), printer, seen);

  printer.Print("\n\n");

  GenerateMessageStatics(descriptor, printer);

  printer.Print("\n\n");

  GenerateMessageXSPackage(descriptor, printer);
}


void
PerlXSGenerator::GenerateMessageModule(const Descriptor* descriptor,
				       OutputDirectory* outdir) const
{
  TProtoStringType filename = descriptor->name() + ".pm";
  std::unique_ptr<io::ZeroCopyOutputStream> output(outdir->Open(filename));
  io::Printer printer(output.get(), '*'); // '*' works well in the .pm file

  std::map<TProtoStringType, TProtoStringType> vars;

  vars["package"] = MessageModuleName(descriptor);
  vars["message"] = descriptor->full_name();
  vars["name"]    = descriptor->name();

  printer.Print(vars,
		"package *package*;\n"
		"\n"
		"use strict;\n"
		"use warnings;\n"
		"use vars qw(@ISA $AUTOLOAD $VERSION);\n"
		"\n"
		"$VERSION = '1.0';\n"
		"\n"
		"use Exporter;\n"
		"\n"
		"require DynaLoader;\n"
		"require AutoLoader;\n"
		"\n"
		"@ISA = qw(DynaLoader Exporter);\n"
		"\n"
		"bootstrap *package* $VERSION;\n"
		"\n"
		"1;\n"
		"\n"
		"__END__\n"
		"\n");
}


void
PerlXSGenerator::GenerateMessagePOD(const Descriptor* descriptor,
				    OutputDirectory* outdir) const
{
  TProtoStringType filename = descriptor->name() + ".pod";
  std::unique_ptr<io::ZeroCopyOutputStream> output(outdir->Open(filename));
  io::Printer printer(output.get(), '*'); // '*' works well in the .pod file

  std::map<TProtoStringType, TProtoStringType> vars;

  vars["package"] = MessageModuleName(descriptor);
  vars["message"] = descriptor->full_name();
  vars["name"]    = descriptor->name();

  // Generate POD documentation for the module.

  printer.Print(vars,
		"=pod\n"
		"\n"
		"=head1 NAME\n"
		"\n"
		"*package* - Perl/XS interface to *message*\n"
		"\n"
		"=head1 SYNOPSIS\n"
		"\n"
		"=head2 Serializing messages\n"
		"\n"
		" #!/usr/bin/perl\n"
		"\n"
		" use strict;\n"
		" use warnings;\n"
		" use *package*;\n"
		"\n"
		" my $*name* = *package*->new;\n"
		" # Set fields in $*name*...\n"
		" my $pack*name* = $*name*->pack();\n"
		"\n"
		"=head2 Unserializing messages\n"
		"\n"
		" #!/usr/bin/perl\n"
		"\n"
		" use strict;\n"
		" use warnings;\n"
		" use *package*;\n"
		"\n"
		" my $pack*name*; # Read this from somewhere...\n"
		" my $*name* = *package*->new;\n"
		" if ( $*name*->unpack($pack*name*) ) {\n"
		"   print \"OK\"\n"
		" } else {\n"
		"   print \"NOT OK\"\n"
		" }\n"
		"\n"
		"=head1 DESCRIPTION\n"
		"\n"
		"*package* defines the following classes:\n"
		"\n"
		"=over 5\n"
		"\n");

  // List of classes

  GenerateDescriptorClassNamePOD(descriptor, printer);

  printer.Print("\n"
		"=back\n"
		"\n");

  GenerateDescriptorMethodPOD(descriptor, printer);

  printer.Print(vars,
		"=head1 AUTHOR\n"
		"\n"
		"Generated from *message* by the protoc compiler.\n"
		"\n"
		"=head1 SEE ALSO\n"
		"\n");

  // Top-level messages in dependency files (recursively expanded)

  printer.Print("http://code.google.com/p/protobuf\n"
		"\n"
		"=cut\n"
		"\n");
}


void
PerlXSGenerator::GenerateDescriptorClassNamePOD(const Descriptor* descriptor,
						io::Printer& printer) const
{
  for ( int i = 0; i < descriptor->enum_type_count(); i++ ) {
    printer.Print("=item C<*name*>\n"
		  "\n"
		  "A wrapper around the *enum* enum\n"
		  "\n",
		  "name", EnumClassName(descriptor->enum_type(i)),
		  "enum", descriptor->enum_type(i)->full_name());
  }

  for ( int i = 0; i < descriptor->nested_type_count(); i++ ) {
    GenerateDescriptorClassNamePOD(descriptor->nested_type(i), printer);
  }

  printer.Print("=item C<*name*>\n"
		"\n"
		"A wrapper around the *message* message\n"
		"\n",
		"name", MessageClassName(descriptor),
		"message", descriptor->full_name());
}


void
PerlXSGenerator::GenerateDescriptorMethodPOD(const Descriptor* descriptor,
					     io::Printer& printer) const
{
  for ( int i = 0; i < descriptor->enum_type_count(); i++ ) {
    const EnumDescriptor * enum_descriptor = descriptor->enum_type(i);
    printer.Print("=head1 C<*name*> values\n"
		  "\n"
		  "=over 4\n"
		  "\n",
		  "name", EnumClassName(enum_descriptor));

    for ( int j = 0; j < enum_descriptor->value_count(); j++ ) {
      PODPrintEnumValue(enum_descriptor->value(j), printer);
    }

    printer.Print("\n"
		  "=back\n"
		  "\n");
  }

  for ( int i = 0; i < descriptor->nested_type_count(); i++ ) {
    GenerateDescriptorMethodPOD(descriptor->nested_type(i), printer);
  }

  // Constructor

  std::map<TProtoStringType, TProtoStringType> vars;

  vars["name"]  = MessageClassName(descriptor);
  vars["value"] = descriptor->name();

  printer.Print(vars,
		"=head1 *name* Constructor\n"
		"\n"
		"=over 4\n"
		"\n"
		"=item B<$*value* = *name*-E<gt>new( [$arg] )>\n"
		"\n"
		"Constructs an instance of C<*name*>.  If a hashref argument\n"
		"is supplied, it is copied into the message instance as if\n"
		"the copy_from() method were called immediately after\n"
		"construction.  Otherwise, if a scalar argument is supplied,\n"
		"it is interpreted as a serialized instance of the message\n"
		"type, and the scalar is parsed to populate the message\n"
		"fields.  Otherwise, if no argument is supplied, an empty\n"
		"message instance is constructed.\n"
		"\n"
		"=back\n"
		"\n"
		"=head1 *name* Methods\n"
		"\n"
		"=over 4\n"
		"\n");

  // Common message methods

  printer.Print(vars,
		"=item B<$*value*2-E<gt>copy_from($*value*1)>\n"
		"\n"
		"Copies the contents of C<*value*1> into C<*value*2>.\n"
		"C<*value*2> is another instance of the same message type.\n"
		"\n"
		"=item B<$*value*2-E<gt>copy_from($hashref)>\n"
		"\n"
		"Copies the contents of C<hashref> into C<*value*2>.\n"
		"C<hashref> is a Data::Dumper-style representation of an\n"
		"instance of the message type.\n"
		"\n"
		"=item B<$*value*2-E<gt>merge_from($*value*1)>\n"
		"\n"
		"Merges the contents of C<*value*1> into C<*value*2>.\n"
		"C<*value*2> is another instance of the same message type.\n"
		"\n"
		"=item B<$*value*2-E<gt>merge_from($hashref)>\n"
		"\n"
		"Merges the contents of C<hashref> into C<*value*2>.\n"
		"C<hashref> is a Data::Dumper-style representation of an\n"
		"instance of the message type.\n"
		"\n"
		"=item B<$*value*-E<gt>clear()>\n"
		"\n"
		"Clears the contents of C<*value*>.\n"
		"\n"
		"=item B<$init = $*value*-E<gt>is_initialized()>\n"
		"\n"
		"Returns 1 if C<*value*> has been initialized with data.\n"
		"\n"
		"=item B<$errstr = $*value*-E<gt>error_string()>\n"
		"\n"
		"Returns a comma-delimited TProtoStringType of initialization errors.\n"
		"\n"
		"=item B<$*value*-E<gt>discard_unknown_fields()>\n"
		"\n"
		"Discards unknown fields from C<*value*>.\n"
		"\n"
		"=item B<$dstr = $*value*-E<gt>debug_string()>\n"
		"\n"
		"Returns a TProtoStringType representation of C<*value*>.\n"
		"\n"
		"=item B<$dstr = $*value*-E<gt>short_debug_string()>\n"
		"\n"
		"Returns a short TProtoStringType representation of C<*value*>.\n"
		"\n"
		"=item B<$ok = $*value*-E<gt>unpack($TProtoStringType)>\n"
		"\n"
		"Attempts to parse C<TProtoStringType> into C<*value*>, returning 1 "
		"on success and 0 on failure.\n"
		"\n"
		"=item B<$TProtoStringType = $*value*-E<gt>pack()>\n"
		"\n"
		"Serializes C<*value*> into C<TProtoStringType>.\n"
		"\n"
		"=item B<$length = $*value*-E<gt>length()>\n"
		"\n"
		"Returns the serialized length of C<*value*>.\n"
		"\n"
		"=item B<@fields = $*value*-E<gt>fields()>\n"
		"\n"
		"Returns the defined fields of C<*value*>.\n"
		"\n"
		"=item B<$hashref = $*value*-E<gt>to_hashref()>\n"
		"\n"
		"Exports the message to a hashref suitable for use in the\n"
		"C<copy_from> or C<merge_from> methods.\n"
		"\n");

  // Message field accessors

  for ( int i = 0; i < descriptor->field_count(); i++ ) {
    const FieldDescriptor* field = descriptor->field(i);

    vars["field"] = field->name();
    vars["type"]  = PODFieldTypeString(field);

    // has_blah or blah_size methods

    if ( field->is_repeated() ) {
      printer.Print(vars,
		    "=item B<$*field*_size = $*value*-E<gt>*field*_size()>\n"
		    "\n"
		    "Returns the number of C<*field*> elements present "
		    "in C<*value*>.\n"
		    "\n");
    } else {
      printer.Print(vars,
		    "=item B<$has_*field* = $*value*-E<gt>has_*field*()>\n"
		    "\n"
		    "Returns 1 if the C<*field*> element of C<*value*> "
		    "is set, 0 otherwise.\n"
		    "\n");
    }

    // clear_blah method

    printer.Print(vars,
		  "=item B<$*value*-E<gt>clear_*field*()>\n"
		  "\n"
		  "Clears the C<*field*> element(s) of C<*value*>.\n"
		  "\n");

    // getters

    if ( field->is_repeated() ) {
      printer.Print(vars,
		    "=item B<@*field*_list = $*value*-E<gt>*field*()>\n"
		    "\n"
		    "Returns all values of C<*field*> in an array.  Each "
		    "element of C<*field*_list> will be *type*.\n"
		    "\n"
		    "=item B<$*field*_elem = $*value*-E<gt>*field*($index)>\n"
		    "\n"
		    "Returns C<*field*> element C<index> from C<*value*>.  "
		    "C<*field*> will be *type*, unless C<index> is out of "
		    "range, in which case it will be undef.\n"
		    "\n");
    } else {
      printer.Print(vars,
		    "=item B<$*field* = $*value*-E<gt>*field*()>\n"
		    "\n"
		    "Returns C<*field*> from C<*value*>.  C<*field*> will "
		    "be *type*.\n"
		    "\n");
    }

    // setters

    if ( field->is_repeated() ) {
      printer.Print(vars,
		    "=item B<$*value*-E<gt>add_*field*($value)>\n"
		    "\n"
		    "Adds C<value> to the list of C<*field*> in C<*value*>.  "
		    "C<value> must be *type*.\n"
		    "\n");
    } else {
      printer.Print(vars,
		    "=item B<$*value*-E<gt>set_*field*($value)>\n"
		    "\n"
		    "Sets the value of C<*field*> in C<*value*> to "
		    "C<value>.  C<value> must be *type*.\n"
		    "\n");
    }
  }

  printer.Print("\n"
		"=back\n"
		"\n");
}


void
PerlXSGenerator::GenerateEnumModule(const EnumDescriptor* enum_descriptor,
				    OutputDirectory* outdir) const
{
  TProtoStringType filename = enum_descriptor->name() + ".pm";
  std::unique_ptr<io::ZeroCopyOutputStream> output(outdir->Open(filename));
  io::Printer printer(output.get(), '*'); // '*' works well in the .pm file

  std::map<TProtoStringType, TProtoStringType> vars;

  vars["package"] = EnumClassName(enum_descriptor);
  vars["enum"]    = enum_descriptor->full_name();

  printer.Print(vars,
		"package *package*;\n"
		"\n"
		"use strict;\n"
		"use warnings;\n"
		"\n");

  // Each enum value is exported as a constant.

  for ( int i = 0; i < enum_descriptor->value_count(); i++ ) {
    std::ostringstream ost;
    ost << enum_descriptor->value(i)->number();
    printer.Print("use constant *value* => *number*;\n",
		  "value", enum_descriptor->value(i)->name(),
		  "number", ost.str().c_str());
  }

  printer.Print("\n"
		"1;\n"
		"\n"
		"__END__\n"
		"\n");

  // Now generate POD for the enum.

  printer.Print(vars,
		"=pod\n"
		"\n"
		"=head1 NAME\n"
		"\n"
		"*package* - Perl interface to *enum*\n"
		"\n"
		"=head1 SYNOPSIS\n"
		"\n"
		" use *package*;\n"
		"\n");

  for ( int i = 0; i < enum_descriptor->value_count(); i++ ) {
    printer.Print(" my $*value* = *package*::*value*;\n",
		  "package", vars["package"],
		  "value", enum_descriptor->value(i)->name());
  }

  printer.Print(vars,
		"\n"
		"=head1 DESCRIPTION\n"
		"\n"
		"*package* defines the following constants:\n"
		"\n"
		"=over 4\n"
		"\n");

  for ( int i = 0; i < enum_descriptor->value_count(); i++ ) {
    PODPrintEnumValue(enum_descriptor->value(i), printer);
  }

  printer.Print(vars,
		"\n"
		"=back\n"
		"\n"
		"=head1 AUTHOR\n"
		"\n"
		"Generated from *enum* by the protoc compiler.\n"
		"\n"
		"=head1 SEE ALSO\n"
		"\n"
		"http://code.google.com/p/protobuf\n"
		"\n"
		"=cut\n"
		"\n");
}


void
PerlXSGenerator::GenerateFileXSTypedefs(const FileDescriptor* file,
					io::Printer& printer,
					std::set<const Descriptor*>& seen) const
{
  for ( int i = 0; i < file->dependency_count(); i++ ) {
    GenerateFileXSTypedefs(file->dependency(i), printer, seen);
  }

  for ( int i = 0; i < file->message_type_count(); i++ ) {
    GenerateMessageXSTypedefs(file->message_type(i), printer, seen);
  }
}


void
PerlXSGenerator::GenerateMessageXSTypedefs(const Descriptor* descriptor,
					   io::Printer& printer,
					   std::set<const Descriptor*>& seen) const
{
  for ( int i = 0; i < descriptor->nested_type_count(); i++ ) {
    GenerateMessageXSTypedefs(descriptor->nested_type(i), printer, seen);
  }

  if ( seen.find(descriptor) == seen.end() ) {
    TProtoStringType cn = cpp::ClassName(descriptor, true);
    TProtoStringType un = StringReplace(cn, "::", "__", true);

    seen.insert(descriptor);
    printer.Print("typedef $classname$ $underscores$;\n",
		  "classname", cn,
		  "underscores", un);
  }
}


void
PerlXSGenerator::GenerateMessageStatics(const Descriptor* descriptor,
					io::Printer& printer) const
{
  for ( int i = 0; i < descriptor->nested_type_count(); i++ ) {
    GenerateMessageStatics(descriptor->nested_type(i), printer);
  }

  std::map<TProtoStringType, TProtoStringType> vars;

  TProtoStringType cn = cpp::ClassName(descriptor, true);
  TProtoStringType un = StringReplace(cn, "::", "__", true);

  vars["depth"]       = "0";
  vars["fieldtype"]   = cn;
  vars["classname"]   = cn;
  vars["underscores"] = un;

  // from_hashref static helper

  printer.Print(vars,
		"static $classname$ *\n"
		"$underscores$_from_hashref ( SV * sv0 )\n"
		"{\n"
		"  $fieldtype$ * msg$depth$ = new $fieldtype$;\n"
		"\n");

  printer.Indent();
  MessageFromHashref(descriptor, printer, vars, 0);
  printer.Outdent();

  printer.Print("\n"
		"  return msg0;\n"
		"}\n"
		"\n");
}


void
PerlXSGenerator::GenerateMessageXSFieldAccessors(const FieldDescriptor* field,
						 io::Printer& printer,
						 const TProtoStringType& classname) const
{
  const Descriptor* descriptor = field->containing_type();
  TProtoStringType            cppname    = cpp::FieldName(field);
  TProtoStringType            perlclass  = MessageClassName(descriptor);
  bool              repeated   = field->is_repeated();

  std::map<TProtoStringType, TProtoStringType> vars;

  vars["classname"] = classname;
  vars["cppname"]   = cppname;
  vars["perlname"]  = field->name();
  vars["perlclass"] = perlclass;

  FieldDescriptor::CppType fieldtype = field->cpp_type();
  FieldDescriptor::Type    type      = field->type();

  if ( fieldtype == FieldDescriptor::CPPTYPE_MESSAGE ) {
    vars["fieldtype"]  = cpp::ClassName(field->message_type(), true);
    vars["fieldclass"] = MessageClassName(field->message_type());
  }

  // For repeated fields, we need an index argument.

  if ( repeated ) {
    vars["i"] = "index";
  } else {
    vars["i"] = "";
  }

  // -------------------------------------------------------------------
  // First, the has_X method or X_size method.
  // -------------------------------------------------------------------

  if ( repeated ) {
    printer.Print(vars,
		  "I32\n"
		  "$perlname$_size(svTHIS)\n"
		  "  SV * svTHIS;\n"
		  "  CODE:\n");
    GenerateTypemapInput(descriptor, printer, "THIS");
    printer.Print(vars,
		  "    RETVAL = THIS->$cppname$_size();\n"
		  "\n"
		  "  OUTPUT:\n"
		  "    RETVAL\n");
  } else if ( HasHasMethod(field) ) {
    printer.Print(vars,
		  "I32\n"
		  "has_$perlname$(svTHIS)\n"
		  "  SV * svTHIS;\n"
		  "  CODE:\n");
    GenerateTypemapInput(descriptor, printer, "THIS");
    printer.Print(vars,
		  "    RETVAL = THIS->has_$cppname$();\n"
		  "\n"
		  "  OUTPUT:\n"
		  "    RETVAL\n");
  }

  printer.Print("\n\n");

  // -------------------------------------------------------------------
  // Next, the "clear" method.
  // -------------------------------------------------------------------

  printer.Print(vars,
		"void\n"
		"clear_$perlname$(svTHIS)\n"
		"  SV * svTHIS;\n"
		"  CODE:\n");
  GenerateTypemapInput(descriptor, printer, "THIS");
  printer.Print(vars,
		"    THIS->clear_$cppname$();\n"
		"\n"
		"\n");

  // -------------------------------------------------------------------
  // Next, the "get" method.
  // -------------------------------------------------------------------

  // Repeated fields have an optional index argument.

  if ( repeated ) {
    printer.Print(vars,
		  "void\n"
		  "$perlname$(svTHIS, ...)\n");
  } else {
    printer.Print(vars,
		  "void\n"
		  "$perlname$(svTHIS)\n");
  }

  printer.Print("  SV * svTHIS;\n"
		"PREINIT:\n"
		"    SV * sv;\n");

  if ( repeated ) {
    printer.Print("    int index = 0;\n");
  }

  // We need to store 64-bit integers as strings in Perl.

  if ( fieldtype == FieldDescriptor::CPPTYPE_INT64 ||
       fieldtype == FieldDescriptor::CPPTYPE_UINT64 ) {
    printer.Print("    std::ostringstream ost;\n");
  }

  if ( fieldtype == FieldDescriptor::CPPTYPE_MESSAGE ) {
    printer.Print(vars,
		  "    $fieldtype$ * val = NULL;\n");
  }

  // We'll use PPCODE in either case, just to make this a little
  // simpler.

  printer.Print("\n"
		"  PPCODE:\n");

  GenerateTypemapInput(descriptor, printer, "THIS");

  // For repeated fields, we need to check the usage ourselves.

  if ( repeated ) {
    printer.Print(vars,
      "    if ( items == 2 ) {\n"
      "      index = SvIV(ST(1));\n"
      "    } else if ( items > 2 ) {\n"
      "      croak(\"Usage: $perlclass$::$perlname$(CLASS, [index])\");\n"
      "    }\n");
  }

  // There are three possibilities now:
  //
  // 1) The user wants a particular element of a repeated field.
  // 2) The user wants all elements of a repeated field.
  // 3) The user wants the value of a non-repeated field.

  if ( repeated ) {
    printer.Print(vars,
		  "    if ( THIS != NULL ) {\n"
		  "      if ( items == 1 ) {\n"
		  "        int count = THIS->$cppname$_size();\n"
		  "\n"
		  "        EXTEND(SP, count);\n"
		  "        for ( int index = 0; index < count; index++ ) {\n");
    PerlSVGetHelper(printer,vars,fieldtype,5);
    printer.Print(vars,
		  "          PUSHs(sv);\n"
		  "        }\n"
		  "      } else if ( index >= 0 &&\n"
		  "                  index < THIS->$cppname$_size() ) {\n"
		  "        EXTEND(SP,1);\n");
    PerlSVGetHelper(printer,vars,fieldtype,4);
    printer.Print("        PUSHs(sv);\n"
		  "      } else {\n"
		  "        EXTEND(SP,1);\n"
		  "        PUSHs(&PL_sv_undef);\n"
		  "      }\n"
		  "    }\n");
  } else {
    printer.Print("    if ( THIS != NULL ) {\n"
		  "      EXTEND(SP,1);\n");
    PerlSVGetHelper(printer,vars,fieldtype,3);
    printer.Print("      PUSHs(sv);\n"
		  "    }\n");
  }

  printer.Print("\n\n");

  // -------------------------------------------------------------------
  // Finally, the "set" method.
  // -------------------------------------------------------------------

  if ( repeated ) {
    printer.Print(vars,
		  "void\n"
		  "add_$perlname$(svTHIS, svVAL)\n");
  } else {
    printer.Print(vars,
		  "void\n"
		  "set_$perlname$(svTHIS, svVAL)\n");
  }

  printer.Print("  SV * svTHIS\n");

  // What is the incoming type?

  switch ( fieldtype ) {
  case FieldDescriptor::CPPTYPE_ENUM:
    vars["etype"] = cpp::ClassName(field->enum_type(), true);
    // Fall through
  case FieldDescriptor::CPPTYPE_INT32:
  case FieldDescriptor::CPPTYPE_BOOL:
    vars["value"] = "svVAL";
    printer.Print("  IV svVAL\n"
		  "\n"
		  "  CODE:\n");
    break;
  case FieldDescriptor::CPPTYPE_UINT32:
    vars["value"] = "svVAL";
    printer.Print("  UV svVAL\n"
		  "\n"
		  "  CODE:\n");
    break;
  case FieldDescriptor::CPPTYPE_FLOAT:
  case FieldDescriptor::CPPTYPE_DOUBLE:
    vars["value"] = "svVAL";
    printer.Print("  NV svVAL\n"
		  "\n"
		  "  CODE:\n");
    break;
  case FieldDescriptor::CPPTYPE_INT64:
    vars["value"] = "lval";
    printer.Print("  char *svVAL\n"
		  "\n"
		  "  PREINIT:\n"
		  "    long long lval;\n"
		  "\n"
		  "  CODE:\n"
		  "    lval = strtoll((svVAL) ? svVAL : \"\", NULL, 0);\n");
    break;
  case FieldDescriptor::CPPTYPE_UINT64:
    vars["value"] = "lval";
    printer.Print("  char *svVAL\n"
		  "\n"
		  "  PREINIT:\n"
		  "    unsigned long long lval;\n"
		  "\n"
		  "  CODE:\n"
		  "    lval = strtoull((svVAL) ? svVAL : \"\", NULL, 0);\n");
    break;
  case FieldDescriptor::CPPTYPE_STRING:
    vars["value"] = "sval";
    printer.Print("  SV *svVAL\n"
		  "\n"
		  "  PREINIT:\n"
		  "    char * str;\n"
		  "    STRLEN len;\n");
    if ( type == FieldDescriptor::TYPE_STRING ) {
      printer.Print(vars,
		    "    TString $value$;\n");
    }
    printer.Print("\n"
		  "  CODE:\n");
    break;
  case FieldDescriptor::CPPTYPE_MESSAGE:
    printer.Print(vars,
		  "  SV * svVAL\n"
		  "  CODE:\n");
    break;
  default:
    vars["value"] = "svVAL";
    break;
  }

  GenerateTypemapInput(descriptor, printer, "THIS");

  if ( fieldtype == FieldDescriptor::CPPTYPE_MESSAGE ) {
    GenerateTypemapInput(field->message_type(), printer, "VAL");
  }

  if ( repeated ) {
    if ( fieldtype == FieldDescriptor::CPPTYPE_MESSAGE ) {
      printer.Print(vars,
		    "    if ( VAL != NULL ) {\n"
		    "      $fieldtype$ * mval = THIS->add_$cppname$();\n"
		    "      mval->CopyFrom(*VAL);\n"
		    "    }\n");
    } else if ( fieldtype == FieldDescriptor::CPPTYPE_ENUM ) {
      printer.Print(vars,
		    "    if ( $etype$_IsValid(svVAL) ) {\n"
		    "      THIS->add_$cppname$(($etype$)svVAL);\n"
		    "    }\n");
    } else if ( fieldtype == FieldDescriptor::CPPTYPE_STRING ) {
      printer.Print("    str = SvPV(svVAL, len);\n");
      if ( type == FieldDescriptor::TYPE_BYTES ) {
	printer.Print(vars,
		      "    THIS->add_$cppname$(str, len);\n");
      } else if ( type == FieldDescriptor::TYPE_STRING ) {
	printer.Print(vars,
		      "    $value$.assign(str, len);\n"
		      "    THIS->add_$cppname$($value$);\n");
      }
    } else {
      printer.Print(vars,
		    "    THIS->add_$cppname$($value$);\n");
    }
  } else {
    if ( fieldtype == FieldDescriptor::CPPTYPE_MESSAGE ) {
      printer.Print(vars,
		    "    if ( VAL != NULL ) {\n"
		    "      $fieldtype$ * mval = THIS->mutable_$cppname$();\n"
		    "      mval->CopyFrom(*VAL);\n"
		    "    }\n");
    } else if ( fieldtype == FieldDescriptor::CPPTYPE_ENUM ) {
      printer.Print(vars,
		    "    if ( $etype$_IsValid(svVAL) ) {\n"
		    "      THIS->set_$cppname$(($etype$)svVAL);\n"
		    "    }\n");
    } else if ( fieldtype == FieldDescriptor::CPPTYPE_STRING ) {
      printer.Print("    str = SvPV(svVAL, len);\n");
      if ( type == FieldDescriptor::TYPE_STRING ) {
	printer.Print(vars,
		      "    sval.assign(str, len);\n"
		      "    THIS->set_$cppname$($value$);\n");
      } else if ( type == FieldDescriptor::TYPE_BYTES ) {
	printer.Print(vars,
		      "    THIS->set_$cppname$(str, len);\n");
      } else {
	// Can't get here
      }
    } else {
      printer.Print(vars,
		    "    THIS->set_$cppname$($value$);\n");
    }
  }

  printer.Print("\n\n");
}


void
PerlXSGenerator::GenerateMessageXSCommonMethods(const Descriptor* descriptor,
						io::Printer& printer,
						const TProtoStringType& classname) const
{
  std::map<TProtoStringType, TProtoStringType> vars;
#if (GOOGLE_PROTOBUF_VERSION >= 2002000)
  FileOptions::OptimizeMode mode;
#endif // GOOGLE_PROTOBUF_VERSION
  TProtoStringType cn = cpp::ClassName(descriptor, true);
  TProtoStringType un = StringReplace(cn, "::", "__", true);

#if (GOOGLE_PROTOBUF_VERSION >= 2002000)
  mode = descriptor->file()->options().optimize_for();
#endif // GOOGLE_PROTOBUF_VERSION

  vars["classname"]   = classname;
  vars["perlclass"]   = MessageClassName(descriptor);
  vars["underscores"] = un;

  // copy_from

  printer.Print(vars,
		"void\n"
		"copy_from(svTHIS, sv)\n"
		"  SV * svTHIS\n"
		"  SV * sv\n"
		"  CODE:\n");
  GenerateTypemapInput(descriptor, printer, "THIS");
  printer.Print(vars,
		"    if ( THIS != NULL && sv != NULL ) {\n"
		"      if ( sv_derived_from(sv, \"$perlclass$\") ) {\n"
		"        IV tmp = SvIV((SV *)SvRV(sv));\n"
		"        $classname$ * other = "
		"INT2PTR($underscores$ *, tmp);\n"
		"\n"
		"        THIS->CopyFrom(*other);\n"
		"      } else if ( SvROK(sv) &&\n"
		"                  SvTYPE(SvRV(sv)) == SVt_PVHV ) {\n"
		"        $classname$ * other = "
		"$underscores$_from_hashref(sv);\n"
		"        THIS->CopyFrom(*other);\n"
		"        delete other;\n"
		"      }\n"
		"    }\n"
		"\n"
		"\n");

  // merge_from

  printer.Print(vars,
		"void\n"
		"merge_from(svTHIS, sv)\n"
		"  SV * svTHIS\n"
		"  SV * sv\n"
		"  CODE:\n");
  GenerateTypemapInput(descriptor, printer, "THIS");
  printer.Print(vars,
		"    if ( THIS != NULL && sv != NULL ) {\n"
		"      if ( sv_derived_from(sv, \"$perlclass$\") ) {\n"
		"        IV tmp = SvIV((SV *)SvRV(sv));\n"
		"        $classname$ * other = "
		"INT2PTR($underscores$ *, tmp);\n"
		"\n"
		"        THIS->MergeFrom(*other);\n"
		"      } else if ( SvROK(sv) &&\n"
		"                  SvTYPE(SvRV(sv)) == SVt_PVHV ) {\n"
		"        $classname$ * other = "
		"$underscores$_from_hashref(sv);\n"
		"        THIS->MergeFrom(*other);\n"
		"        delete other;\n"
		"      }\n"
		"    }\n"
		"\n"
		"\n");

  // clear

  printer.Print(vars,
		"void\n"
		"clear(svTHIS)\n"
		"  SV * svTHIS\n"
		"  CODE:\n");
  GenerateTypemapInput(descriptor, printer, "THIS");
  printer.Print(vars,
		"    if ( THIS != NULL ) {\n"
		"      THIS->Clear();\n"
		"    }\n"
		"\n"
		"\n");

  // is_initialized

  printer.Print(vars,
		"int\n"
		"is_initialized(svTHIS)\n"
		"  SV * svTHIS\n"
		"  CODE:\n");
  GenerateTypemapInput(descriptor, printer, "THIS");
  printer.Print(vars,
		"    if ( THIS != NULL ) {\n"
		"      RETVAL = THIS->IsInitialized();\n"
		"    } else {\n"
		"      RETVAL = 0;\n"
		"    }\n"
		"\n"
		"  OUTPUT:\n"
		"    RETVAL\n"
		"\n"
		"\n");

  // error_string

  printer.Print(vars,
		"SV *\n"
		"error_string(svTHIS)\n"
		"  SV * svTHIS\n"
		"  PREINIT:\n"
		"    TString estr;\n"
		"\n"
		"  CODE:\n");
  GenerateTypemapInput(descriptor, printer, "THIS");
  printer.Print(vars,
		"    if ( THIS != NULL ) {\n"
		"      estr = THIS->InitializationErrorString();\n"
		"    }\n"
		"    RETVAL = newSVpv(estr.c_str(), estr.length());\n"
		"\n"
		"  OUTPUT:\n"
		"    RETVAL\n"
		"\n"
		"\n");

  // LITE_RUNTIME does not include certain methods.

#if (GOOGLE_PROTOBUF_VERSION >= 2002000)
  if (mode != FileOptions::LITE_RUNTIME) {
#endif // GOOGLE_PROTOBUF_VERSION

    // discard_unknown_fields

    printer.Print(vars,
		  "void\n"
		  "discard_unkown_fields(svTHIS)\n"
		  "  SV * svTHIS\n"
		  "  CODE:\n");
    GenerateTypemapInput(descriptor, printer, "THIS");
    printer.Print(vars,
		  "    if ( THIS != NULL ) {\n"
		  "      THIS->DiscardUnknownFields();\n"
		  "    }\n"
		  "\n"
		  "\n");

    // debug_string

    printer.Print(vars,
		  "SV *\n"
		  "debug_string(svTHIS)\n"
		  "  SV * svTHIS\n"
		  "  PREINIT:\n"
		  "    TString dstr;\n"
		  "\n"
		  "  CODE:\n");
    GenerateTypemapInput(descriptor, printer, "THIS");
    printer.Print(vars,
		  "    if ( THIS != NULL ) {\n"
		  "      dstr = THIS->DebugString();\n"
		  "    }\n"
		  "    RETVAL = newSVpv(dstr.c_str(), dstr.length());\n"
		  "\n"
		  "  OUTPUT:\n"
		  "    RETVAL\n"
		  "\n"
		  "\n");

    // short_debug_string

    printer.Print(vars,
		  "SV *\n"
		  "short_debug_string(svTHIS)\n"
		  "  SV * svTHIS\n"
		  "  PREINIT:\n"
		  "    TString dstr;\n"
		  "\n"
		  "  CODE:\n");
    GenerateTypemapInput(descriptor, printer, "THIS");
    printer.Print(vars,
		  "    if ( THIS != NULL ) {\n"
		  "      dstr = THIS->ShortDebugString();\n"
		  "    }\n"
		  "    RETVAL = newSVpv(dstr.c_str(), dstr.length());\n"
		  "\n"
		  "  OUTPUT:\n"
		  "    RETVAL\n"
		  "\n"
		  "\n");
#if (GOOGLE_PROTOBUF_VERSION >= 2002000)
  }
#endif // GOOGLE_PROTOBUF_VERSION

  // unpack

  printer.Print(vars,
		"int\n"
		"unpack(svTHIS, arg)\n"
		"  SV * svTHIS\n"
		"  SV * arg\n"
		"  PREINIT:\n"
		"    STRLEN len;\n"
		"    char * str;\n"
		"\n"
		"  CODE:\n");
  GenerateTypemapInput(descriptor, printer, "THIS");
  printer.Print(vars,
		"    if ( THIS != NULL ) {\n"
		"      str = SvPV(arg, len);\n"
		"      if ( str != NULL ) {\n"
		"        RETVAL = THIS->ParseFromArray(str, len);\n"
		"      } else {\n"
		"        RETVAL = 0;\n"
		"      }\n"
		"    } else {\n"
		"      RETVAL = 0;\n"
		"    }\n"
		"\n"
		"  OUTPUT:\n"
		"    RETVAL\n"
		"\n"
		"\n");

  // pack

  printer.Print(vars,
		"SV *\n"
		"pack(svTHIS)\n"
		"  SV * svTHIS\n");

  // This may be controlled by a custom option at some point.
#if NO_ZERO_COPY
  printer.Print(vars,
		"  PREINIT:\n"
		"    TProtoStringType output;\n"
		"\n");
#endif

  printer.Print(vars,
		"  CODE:\n");
  GenerateTypemapInput(descriptor, printer, "THIS");
  printer.Print(vars,
		"    if ( THIS != NULL ) {\n");

  vars["base"] = cpp::StripProto(descriptor->file()->name());

  printer.Print(vars,
		"      RETVAL = newSVpvn(\"\", 0);\n"
		"      $base$_OutputStream os(RETVAL);\n"
		"      if ( THIS->IsInitialized() ) {\n"
		"        if ( THIS->SerializePartialToZeroCopyStream(&os)"
		"!= true ) {\n"
		"          SvREFCNT_dec(RETVAL);\n"
		"          RETVAL = Nullsv;\n"
		"        } else {\n"
		"          os.Sync();\n"
		"        }\n"
		"      } else {\n"
		"        croak(\"Can't serialize message of type "
		"'$perlclass$' because it is missing required fields: %s\",\n"
		"              THIS->InitializationErrorString().c_str());\n"
		"      }\n");

  printer.Print(vars,
		"    } else {\n"
		"      RETVAL = Nullsv;\n"
		"    }\n"
		"\n"
		"  OUTPUT:\n"
		"    RETVAL\n"
		"\n"
		"\n");

  // length

  printer.Print(vars,
		"int\n"
		"length(svTHIS)\n"
		"  SV * svTHIS\n"
		"  CODE:\n");
  GenerateTypemapInput(descriptor, printer, "THIS");
  printer.Print(vars,
		"    if ( THIS != NULL ) {\n"
		"      RETVAL = THIS->ByteSize();\n"
		"    } else {\n"
		"      RETVAL = 0;\n"
		"    }\n"
		"\n"
		"  OUTPUT:\n"
		"    RETVAL\n"
		"\n"
		"\n");

  // fields

  std::ostringstream field_count;

  field_count << descriptor->field_count();
  vars["field_count"] = field_count.str();
  printer.Print(vars,
		"void\n"
		"fields(svTHIS)\n"
		"  SV * svTHIS\n"
		"  PPCODE:\n"
		"    (void)svTHIS;\n"
		"    EXTEND(SP, $field_count$);\n");

  for ( int i = 0; i < descriptor->field_count(); i++ ) {
    const FieldDescriptor* field = descriptor->field(i);
    vars["field"] = field->name();
    printer.Print(vars,
		  "    PUSHs(sv_2mortal(newSVpv(\"$field$\",0)));\n"
		  );
  }

  printer.Print("\n\n");

  // to_hashref

  printer.Print(vars,
		"SV *\n"
		"to_hashref(svTHIS)\n"
		"  SV * svTHIS\n"
		"  CODE:\n");
  GenerateTypemapInput(descriptor, printer, "THIS");
  printer.Print(vars,
		"    if ( THIS != NULL ) {\n"
		"      HV * hv0 = newHV();\n"
		"      $classname$ * msg0 = THIS;\n"
		"\n");

  vars["depth"] = "0";
  vars["fieldtype"] = classname;

  printer.Indent();
  printer.Indent();
  printer.Indent();
  MessageToHashref(descriptor, printer, vars, 0);
  printer.Outdent();
  printer.Outdent();
  printer.Outdent();

  printer.Print("      RETVAL = newRV_noinc((SV *)hv0);\n"
		"    } else {\n"
		"      RETVAL = Nullsv;\n"
		"    }\n"
		"\n"
		"  OUTPUT:\n"
		"    RETVAL\n"
		"\n"
		"\n");
}

void
PerlXSGenerator::GenerateMessageXSPackage(const Descriptor* descriptor,
					  io::Printer& printer) const
{
  for ( int i = 0; i < descriptor->nested_type_count(); i++ ) {
    GenerateMessageXSPackage(descriptor->nested_type(i), printer);
  }

  std::map<TProtoStringType, TProtoStringType> vars;

  TProtoStringType cn = cpp::ClassName(descriptor, true);
  TProtoStringType mn = MessageModuleName(descriptor);
  TProtoStringType pn = MessageClassName(descriptor);
  TProtoStringType un = StringReplace(cn, "::", "__", true);

  vars["module"]      = mn;
  vars["classname"]   = cn;
  vars["package"]     = pn;
  vars["underscores"] = un;

  printer.Print(vars,
		"MODULE = $module$ PACKAGE = $package$\n"
		"PROTOTYPES: ENABLE\n"
		"\n"
		"\n");

  // BOOT (if there are enum types)

  int enum_count = descriptor->enum_type_count();

  if ( enum_count > 0 ) {
    printer.Print("BOOT:\n"
		  "  {\n"
		  "    HV * stash;\n\n");

    printer.Indent();
    printer.Indent();
    for ( int i = 0; i < enum_count; i++ ) {
      const EnumDescriptor * etype = descriptor->enum_type(i);
      int vcount = etype->value_count();

      printer.Print("stash = gv_stashpv(\"$package$::$name$\", TRUE);\n",
		    "package", pn,
		    "name", etype->name());
      for ( int j = 0; j < vcount; j++ ) {
	const EnumValueDescriptor * vdesc = etype->value(j);
	printer.Print(
	  "newCONSTSUB(stash, \"$name$\", newSViv($classname$::$name$));\n",
	  "classname", cn,
	  "name", vdesc->name()
	);
      }
    }
    printer.Outdent();
    printer.Outdent();
    printer.Print("  }\n\n\n");
  }

  // Constructor

  printer.Print(vars,
		"SV *\n"
		"$classname$::new (...)\n"
		"  PREINIT:\n"
		"    $classname$ * rv = NULL;\n"
		"\n"
		"  CODE:\n"
		"    if ( strcmp(CLASS,\"$package$\") ) {\n"
		"      croak(\"invalid class %s\",CLASS);\n"
		"    }\n"
		"    if ( items == 2 && ST(1) != Nullsv ) {\n"
		"      if ( SvROK(ST(1)) && "
		"SvTYPE(SvRV(ST(1))) == SVt_PVHV ) {\n"
		"        rv = $underscores$_from_hashref(ST(1));\n"
		"      } else {\n"
		"        STRLEN len;\n"
		"        char * str;\n"
		"\n"
		"        rv = new $classname$;\n"
		"        str = SvPV(ST(1), len);\n"
		"        if ( str != NULL ) {\n"
		"          rv->ParseFromArray(str, len);\n"
		"        }\n"
		"      }\n"
		"    } else {\n"
		"      rv = new $classname$;\n"
		"    }\n"
		"    RETVAL = newSV(0);\n"
		"    sv_setref_pv(RETVAL, \"$package$\", (void *)rv);\n"
		"\n"
		"  OUTPUT:\n"
		"    RETVAL\n"
		"\n"
		"\n");

  // Destructor

  printer.Print(vars,
		"void\n"
		"DESTROY(svTHIS)\n"
		"  SV * svTHIS;\n"
		"  CODE:\n");
  GenerateTypemapInput(descriptor, printer, "THIS");
  printer.Print("    if ( THIS != NULL ) {\n"
		"      delete THIS;\n"
		"    }\n"
		"\n"
		"\n");

  // Message methods (copy_from, parse_from, etc).

  GenerateMessageXSCommonMethods(descriptor, printer, cn);

  // Field accessors

  for ( int i = 0; i < descriptor->field_count(); i++ ) {
    GenerateMessageXSFieldAccessors(descriptor->field(i), printer, cn);
  }
}


void
PerlXSGenerator::GenerateTypemapInput(const Descriptor* descriptor,
				      io::Printer& printer,
				      const TProtoStringType& svname) const
{
  std::map<TProtoStringType, TProtoStringType> vars;

  TProtoStringType cn = cpp::ClassName(descriptor, true);

  vars["classname"]   = cn;
  vars["perlclass"]   = MessageClassName(descriptor);
  vars["underscores"] = StringReplace(cn, "::", "__", true);
  vars["svname"]      = svname;

  printer.Print(vars,
		"    $classname$ * $svname$;\n"
		"    if ( sv_derived_from(sv$svname$, \"$perlclass$\") ) {\n"
		"      IV tmp = SvIV((SV *)SvRV(sv$svname$));\n"
		"      $svname$ = INT2PTR($underscores$ *, tmp);\n"
		"    } else {\n"
		"      croak(\"$svname$ is not of type $perlclass$\");\n"
		"    }\n");
}

// Returns the containing Perl module name for a message descriptor.

TProtoStringType
PerlXSGenerator::MessageModuleName(const Descriptor* descriptor) const
{
  const Descriptor *container = descriptor;

  while (container->containing_type() != NULL) {
    container = container->containing_type();
  }

  return MessageClassName(container);
}

// Returns the Perl class name for a message descriptor.

TProtoStringType
PerlXSGenerator::MessageClassName(const Descriptor* descriptor) const
{
  return PackageName(descriptor->full_name(), descriptor->file()->package());
}

// Returns the Perl class name for a message descriptor.

TProtoStringType
PerlXSGenerator::EnumClassName(const EnumDescriptor* descriptor) const
{
  return PackageName(descriptor->full_name(), descriptor->file()->package());
}

// Possibly replace the package prefix with the --perlxs-package value

TProtoStringType
PerlXSGenerator::PackageName(const TProtoStringType& name, const TProtoStringType& package) const
{
  TProtoStringType output;

  if (!perlxs_package_.empty()) {
    output = StringReplace(name, package.c_str(), perlxs_package_.c_str(), false);
    output = StringReplace(output, ".", "::", true);
  } else {
    output = StringReplace(name, ".", "::", true);
  }

  return output;
}

void
PerlXSGenerator::PerlSVGetHelper(io::Printer& printer,
				 const std::map<TProtoStringType, TProtoStringType>& vars,
				 FieldDescriptor::CppType fieldtype,
				 int depth) const
{
  for ( int i = 0; i < depth; i++ ) {
    printer.Indent();
  }

  switch ( fieldtype ) {
  case FieldDescriptor::CPPTYPE_INT32:
  case FieldDescriptor::CPPTYPE_BOOL:
  case FieldDescriptor::CPPTYPE_ENUM:
    printer.Print(vars,
		  "sv = sv_2mortal(newSViv(THIS->$cppname$($i$)));\n");
    break;
  case FieldDescriptor::CPPTYPE_UINT32:
    printer.Print(vars,
		  "sv = sv_2mortal(newSVuv(THIS->$cppname$($i$)));\n");
    break;
  case FieldDescriptor::CPPTYPE_FLOAT:
  case FieldDescriptor::CPPTYPE_DOUBLE:
    printer.Print(vars,
		  "sv = sv_2mortal(newSVnv(THIS->$cppname$($i$)));\n");
    break;
  case FieldDescriptor::CPPTYPE_INT64:
  case FieldDescriptor::CPPTYPE_UINT64:
    printer.Print(vars,
		  "ost.str(\"\");\n"
		  "ost << THIS->$cppname$($i$);\n"
		  "sv = sv_2mortal(newSVpv(ost.str().c_str(),\n"
		  "                        ost.str().length()));\n");
    break;
  case FieldDescriptor::CPPTYPE_STRING:
    printer.Print(vars,
		  "sv = sv_2mortal(newSVpv(THIS->$cppname$($i$).c_str(),\n"
		  "                        "
		  "THIS->$cppname$($i$).length()));\n");
    break;
  case FieldDescriptor::CPPTYPE_MESSAGE:
    printer.Print(vars,
		  "val = new $fieldtype$;\n"
		  "val->CopyFrom(THIS->$cppname$($i$));\n"
		  "sv = sv_newmortal();\n"
		  "sv_setref_pv(sv, \"$fieldclass$\", (void *)val);\n");
    break;
  default:
    printer.Print("sv = &PL_sv_undef;\n");
    break;
  }

  for ( int i = 0; i < depth; i++ ) {
    printer.Outdent();
  }
}

void
PerlXSGenerator::PODPrintEnumValue(const EnumValueDescriptor *value,
				   io::Printer& printer) const
{
  std::ostringstream ost;
  printer.Print("=item B<*value*>\n"
		"\n",
		"value", value->name());
  ost << value->number();
  printer.Print("This constant has a value of *number*.\n"
		"\n",
		"number", ost.str().c_str());
}

TProtoStringType
PerlXSGenerator::PODFieldTypeString(const FieldDescriptor* field) const
{
  TProtoStringType type;

  switch ( field->cpp_type() ) {
  case FieldDescriptor::CPPTYPE_INT32:
    type = "a 32-bit signed integer";
    break;
  case FieldDescriptor::CPPTYPE_BOOL:
    type = "a Boolean value";
    break;
  case FieldDescriptor::CPPTYPE_ENUM:
    type = "a value of " + EnumClassName(field->enum_type());
    break;
  case FieldDescriptor::CPPTYPE_UINT32:
    type = "a 32-bit unsigned integer";
    break;
  case FieldDescriptor::CPPTYPE_FLOAT:
  case FieldDescriptor::CPPTYPE_DOUBLE:
    type = "a floating point number";
    break;
  case FieldDescriptor::CPPTYPE_INT64:
    type = "a 64-bit signed integer";
    break;
  case FieldDescriptor::CPPTYPE_UINT64:
    type = "a 64-bit unsigned integer";
    break;
  case FieldDescriptor::CPPTYPE_STRING:
    type = "a TProtoStringType";
    break;
  case FieldDescriptor::CPPTYPE_MESSAGE:
    type = "an instance of " + MessageClassName(field->message_type());
    break;
  default:
    type = "an unknown type";
    break;
  }

  return type;
}

void
PerlXSGenerator::StartFieldToHashref(const FieldDescriptor * field,
				     io::Printer& printer,
				     std::map<TProtoStringType, TProtoStringType>& vars,
				     int depth) const
{
  SetupDepthVars(vars, depth);

  if ( field->is_repeated() ) {
    vars["i"] = "i" + vars["pdepth"];
    printer.Print(vars,
		  "if ( msg$pdepth$->$cppname$_size() > 0 ) {\n");
    printer.Indent();
    printer.Print(vars,
		  "AV * av$pdepth$ = newAV();\n"
		  "SV * sv$pdepth$ = newRV_noinc((SV *)av$pdepth$);\n"
		  "\n"
		  "for ( int $i$ = 0; "
		  "$i$ < msg$pdepth$->$cppname$_size(); $i$++ ) {\n");
  } else {
    vars["i"] = "";
    if ( HasHasMethod(field) ) {
        printer.Print(vars,
              "if ( msg$pdepth$->has_$cppname$() ) {\n");
    }
    else {
        printer.Print(vars,
              "{\n");
    }
  }
  printer.Indent();
}

void
PerlXSGenerator::FieldToHashrefHelper(io::Printer& printer,
				      std::map<TProtoStringType, TProtoStringType>& vars,
				      const FieldDescriptor* field) const
{
  vars["msg"] = "msg" + vars["pdepth"];
  if ( field->is_repeated() ) {
    vars["sv"] = "sv" + vars["depth"];
  } else {
    vars["sv"] = "sv" + vars["pdepth"];
  }

  switch ( field->cpp_type() ) {
  case FieldDescriptor::CPPTYPE_INT32:
  case FieldDescriptor::CPPTYPE_BOOL:
  case FieldDescriptor::CPPTYPE_ENUM:
    printer.Print(vars,
		  "SV * $sv$ = newSViv($msg$->$cppname$($i$));\n");
    break;
  case FieldDescriptor::CPPTYPE_UINT32:
    printer.Print(vars,
		  "SV * $sv$ = newSVuv($msg$->$cppname$($i$));\n");
    break;
  case FieldDescriptor::CPPTYPE_FLOAT:
  case FieldDescriptor::CPPTYPE_DOUBLE:
    printer.Print(vars,
		  "SV * $sv$ = newSVnv($msg$->$cppname$($i$));\n");
    break;
  case FieldDescriptor::CPPTYPE_INT64:
  case FieldDescriptor::CPPTYPE_UINT64:
    printer.Print(vars,
		  "std::ostringstream ost$pdepth$;\n"
		  "\n"
		  "ost$pdepth$ << $msg$->$cppname$($i$);\n"
		  "SV * $sv$ = newSVpv(ost$pdepth$.str().c_str(),"
		  " ost$pdepth$.str().length());\n");
    break;
  case FieldDescriptor::CPPTYPE_STRING:
  case FieldDescriptor::CPPTYPE_MESSAGE:
  default:
    printer.Print(vars,
		  "SV * $sv$ = newSVpv($msg$->"
		  "$cppname$($i$).c_str(), $msg$->"
		  "$cppname$($i$).length());\n");
    break;
  }
}

void
PerlXSGenerator::EndFieldToHashref(const FieldDescriptor * field,
				   io::Printer& printer,
				   std::map<TProtoStringType, TProtoStringType>& vars,
				   int depth) const
{
  vars["field"] = field->name();

  SetupDepthVars(vars, depth);

  if ( field->is_repeated() ) {
    printer.Print(vars,
		  "av_push(av$pdepth$, sv$depth$);\n");
    printer.Outdent();
    printer.Print(vars,
		  "}\n"
		  "hv_store(hv$pdepth$, \"$field$\", "
		  "sizeof(\"$field$\") - 1, sv$pdepth$, 0);\n");
  } else {
    if ( field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE ) {
      printer.Print(vars,
		    "hv_store(hv$pdepth$, \"$field$\", "
		    "sizeof(\"$field$\") - 1, sv$depth$, 0);\n");
    } else {
      printer.Print(vars,
		    "hv_store(hv$pdepth$, \"$field$\", "
		    "sizeof(\"$field$\") - 1, sv$pdepth$, 0);\n");
    }
  }
  printer.Outdent();
  printer.Print("}\n");
}

void
PerlXSGenerator::MessageToHashref(const Descriptor * descriptor,
				  io::Printer& printer,
				  std::map<TProtoStringType, TProtoStringType>& vars,
				  int depth) const
{
  int i;

  // Iterate the fields

  for ( i = 0; i < descriptor->field_count(); i++ ) {
    const FieldDescriptor* field = descriptor->field(i);
    FieldDescriptor::CppType fieldtype = field->cpp_type();

    vars["field"] = field->name();
    vars["cppname"] = cpp::FieldName(field);

    StartFieldToHashref(field, printer, vars, depth);

    if ( fieldtype == FieldDescriptor::CPPTYPE_MESSAGE ) {
      vars["fieldtype"] = cpp::ClassName(field->message_type(), true);
      printer.Print(vars,
		    "$fieldtype$ * msg$ndepth$ = msg$pdepth$->"
		    "mutable_$cppname$($i$);\n"
		    "HV * hv$ndepth$ = newHV();\n"
		    "SV * sv$depth$ = newRV_noinc((SV *)hv$ndepth$);\n"
		    "\n");
      MessageToHashref(field->message_type(), printer, vars, depth + 2);
      SetupDepthVars(vars, depth);
    } else {
      FieldToHashrefHelper(printer, vars, field);
    }

    EndFieldToHashref(field, printer, vars, depth);
  }
}

void
PerlXSGenerator::FieldFromHashrefHelper(io::Printer& printer,
					std::map<TProtoStringType, TProtoStringType>& vars,
					const FieldDescriptor * field) const
{
  vars["msg"] = "msg" + vars["pdepth"];
  vars["var"] = "*sv" + vars["depth"];

  if ( field->is_repeated() ) {
    vars["do"]  = "add";
  } else {
    vars["do"]  = "set";
  }

  switch ( field->cpp_type() ) {
  case FieldDescriptor::CPPTYPE_INT32:
  case FieldDescriptor::CPPTYPE_BOOL:
    printer.Print(vars,
		  "$msg$->$do$_$cppname$(SvIV($var$));\n");
    break;
  case FieldDescriptor::CPPTYPE_ENUM:
    vars["etype"] = cpp::ClassName(field->enum_type(), true);
    printer.Print(vars,
		  "$msg$->$do$_$cppname$"
		  "(($etype$)SvIV($var$));\n");
    break;
  case FieldDescriptor::CPPTYPE_UINT32:
    printer.Print(vars,
		  "$msg$->$do$_$cppname$(SvUV($var$));\n");
    break;
  case FieldDescriptor::CPPTYPE_FLOAT:
  case FieldDescriptor::CPPTYPE_DOUBLE:
    printer.Print(vars,
		  "$msg$->$do$_$cppname$(SvNV($var$));\n");
    break;
  case FieldDescriptor::CPPTYPE_INT64:
    printer.Print(vars,
		  "arc_i64 iv$pdepth$ = "
		  "strtoll(SvPV_nolen($var$), NULL, 0);\n"
		  "\n"
		  "$msg$->$do$_$cppname$(iv$pdepth$);\n");
    break;
  case FieldDescriptor::CPPTYPE_UINT64:
    printer.Print(vars,
		  "arc_ui64 uv$pdepth$ = "
		  "strtoull(SvPV_nolen($var$), NULL, 0);\n"
		  "\n"
		  "$msg$->$do$_$cppname$(uv$pdepth$);\n");
    break;
  case FieldDescriptor::CPPTYPE_STRING:
    printer.Print("STRLEN len;\n"
		  "char * str;\n");

    if ( field->type() == FieldDescriptor::TYPE_STRING ) {
      printer.Print("TString sval;\n");
    }

    printer.Print(vars,
		  "\n"
		  "str = SvPV($var$, len);\n");

    if ( field->type() == FieldDescriptor::TYPE_STRING ) {
      printer.Print(vars,
		    "sval.assign(str, len);\n"
		    "$msg$->$do$_$cppname$(sval);\n");
    } else if ( field->type() == FieldDescriptor::TYPE_BYTES ) {
      printer.Print(vars,
		    "$msg$->$do$_$cppname$(str, len);\n");
    } else {
      // Can't get here
    }
    break;
  case FieldDescriptor::CPPTYPE_MESSAGE:
  // Should never get here.
  default:
    break;
  }
}

void
PerlXSGenerator::MessageFromHashref(const Descriptor * descriptor,
				    io::Printer& printer,
				    std::map<TProtoStringType, TProtoStringType>& vars,
				    int depth) const
{
  int i;

  SetupDepthVars(vars, depth);

  printer.Print(vars,
		"if ( SvROK(sv$pdepth$) && "
		"SvTYPE(SvRV(sv$pdepth$)) == SVt_PVHV ) {\n");
  printer.Indent();
  printer.Print(vars,
		"HV *  hv$pdepth$ = (HV *)SvRV(sv$pdepth$);\n"
		"SV ** sv$depth$;\n"
		"\n");

  // Iterate the fields

  for ( i = 0; i < descriptor->field_count(); i++ ) {
    const FieldDescriptor* field = descriptor->field(i);

    vars["field"]   = field->name();
    vars["cppname"] = cpp::FieldName(field);

    if ( field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE ) {
      vars["fieldtype"] = cpp::ClassName(field->message_type(), true);
    }

    printer.Print(vars,
		  "if ( (sv$depth$ = hv_fetch(hv$pdepth$, "
		  "\"$field$\", sizeof(\"$field$\") - 1, 0)) != NULL ) {\n");

    printer.Indent();

    if ( field->is_repeated() ) {
      printer.Print(vars,
		    "if ( SvROK(*sv$depth$) && "
		    "SvTYPE(SvRV(*sv$depth$)) == SVt_PVAV ) {\n");
      printer.Indent();
      printer.Print(vars,
		    "AV * av$depth$ = (AV *)SvRV(*sv$depth$);\n"
		    "\n"
		    "for ( int i$depth$ = 0; "
		    "i$depth$ <= av_len(av$depth$); i$depth$++ ) {\n");
      printer.Indent();

      if ( field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE ) {
	printer.Print(vars,
		      "$fieldtype$ * msg$ndepth$ = "
		      "msg$pdepth$->add_$cppname$();\n"
		      "SV ** sv$depth$;\n"
		      "SV *  sv$ndepth$;\n"
		      "\n"
		      "if ( (sv$depth$ = "
		      "av_fetch(av$depth$, i$depth$, 0)) != NULL ) {\n"
		      "  sv$ndepth$ = *sv$depth$;\n");
      } else {
	printer.Print(vars,
		      "SV ** sv$depth$;\n"
		      "\n"
		      "if ( (sv$depth$ = "
		      "av_fetch(av$depth$, i$depth$, 0)) != NULL ) {\n");
      }
      printer.Indent();
    } else {
      if ( field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE ) {
	printer.Print(vars,
		      "$fieldtype$ * msg$ndepth$ = "
		      "msg$pdepth$->mutable_$cppname$();\n"
		      "SV * sv$ndepth$ = *sv$depth$;\n"
		      "\n");
      }
    }

    if ( field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE ) {
      MessageFromHashref(field->message_type(), printer, vars, depth + 2);
      SetupDepthVars(vars, depth);
    } else {
      FieldFromHashrefHelper(printer, vars, field);
    }

    if ( field->is_repeated() ) {
      printer.Outdent();
      printer.Print("}\n");
      printer.Outdent();
      printer.Print("}\n");
      printer.Outdent();
      printer.Print("}\n");
    }

    printer.Outdent();
    printer.Print("}\n");
  }

  printer.Outdent();
  printer.Print("}\n");
}

}  // namespace perlxs
}  // namespace compiler
}  // namespace protobuf
}  // namespace google
