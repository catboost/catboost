/* ----------------------------------------------------------------------------
 * This file is part of SWIG, which is licensed as a whole under version 3
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at https://www.swig.org/legal.html.
 *
 * scilab.cxx
 *
 * Scilab language module for SWIG.
 * --------------------------------------------------------------------------*/

#include "swigmod.h"
#include <cstddef>
#include <cstdlib>

static const int SCILAB_IDENTIFIER_NAME_CHAR_MAX = 24;

static const char *usage = (char *) " \
Scilab options (available with -scilab)\n \
     -builder                               - Generate a Scilab builder script\n \
     -buildercflags <cflags>                - Add <cflags> to the builder compiler flags\n \
     -builderflagscript <file>              - Set the Scilab script <file> to use by builder to configure the build flags\n \
     -builderldflags <ldflags>              - Add <ldflags> to the builder linker flags\n \
     -buildersources <files>                - Add the (comma separated) files <files> to the builder sources\n \
     -builderverbositylevel <level>         - Set the builder verbosity level to <level> (default 0: off, 2: high)\n \
     -gatewayxml <gateway_id>               - Generate gateway xml with the given <gateway_id>\n \
     -gatewayxml6                           - Generate gateway xml for Scilab 6\n \
\n";


class SCILAB:public Language {
protected:
  /* General objects used for holding the strings */
  File *beginSection;
  File *runtimeSection;
  File *headerSection;
  File *wrappersSection;
  File *initSection;

  String *variablesCode;

  bool generateBuilder;
  File *builderFile;
  String *builderCode;
  String *builderCode5;
  String *builderCode6;
  int builderFunctionCount;

  List *sourceFileList;
  List *cflags;
  List *ldflags;

  String *verboseBuildLevel;
  String *buildFlagsScript;

  String *gatewayHeader;
  String *gatewayHeaderV5;
  String *gatewayHeaderV6;

  bool createGatewayXML;
  File *gatewayXMLFile;
  String *gatewayXML;
  String *gatewayID;
  int primitiveID;

  bool createGatewayXMLV6;
  File *gatewayXMLFileV6;
  String *gatewayXMLV6;

  bool createLoader;
  File *loaderFile;
  String *loaderScript;
  String *loaderScript5;
  String *loaderScript6;
  int loaderFunctionCount;
public:

  /* ------------------------------------------------------------------------
   * main()
   * ----------------------------------------------------------------------*/

  virtual void main(int argc, char *argv[]) {
    generateBuilder = false;
    sourceFileList = NewList();
    cflags = NewList();
    ldflags = NewList();
    verboseBuildLevel = NULL;
    buildFlagsScript = NULL;

    gatewayHeader = NULL;
    gatewayHeaderV5 = NULL;
    gatewayHeaderV6 = NULL;

    createGatewayXML = false;
    gatewayXML = NULL;
    gatewayXMLFile = NULL;
    gatewayID = NULL;

    createGatewayXMLV6 = false;
    gatewayXMLV6 = NULL;
    gatewayXMLFileV6 = NULL;

    createLoader = true;
    loaderFile = NULL;
    loaderScript = NULL;

    /* Manage command line arguments */
    for (int argIndex = 1; argIndex < argc; argIndex++) {
      if (argv[argIndex] != NULL) {
	if (strcmp(argv[argIndex], "-help") == 0) {
	  Printf(stdout, "%s\n", usage);
	} else if (strcmp(argv[argIndex], "-builder") == 0) {
	  Swig_mark_arg(argIndex);
	  generateBuilder = true;
	  createLoader = false;
	} else if (strcmp(argv[argIndex], "-buildersources") == 0) {
	  if (argv[argIndex + 1] != NULL) {
	    Swig_mark_arg(argIndex);
	    char *sourceFile = strtok(argv[argIndex + 1], ",");
	    while (sourceFile != NULL) {
	      Insert(sourceFileList, Len(sourceFileList), sourceFile);
	      sourceFile = strtok(NULL, ",");
	    }
	    Swig_mark_arg(argIndex + 1);
	  }
	} else if (strcmp(argv[argIndex], "-buildercflags") == 0) {
          Swig_mark_arg(argIndex);
          if (argv[argIndex + 1] != NULL) {
            Insert(cflags, Len(cflags), argv[argIndex + 1]);
            Swig_mark_arg(argIndex + 1);
          }
        } else if (strcmp(argv[argIndex], "-builderldflags") == 0) {
          Swig_mark_arg(argIndex);
          if (argv[argIndex + 1] != NULL) {
            Insert(ldflags, Len(ldflags), argv[argIndex + 1]);
            Swig_mark_arg(argIndex + 1);
          }
        } else if (strcmp(argv[argIndex], "-builderverbositylevel") == 0) {
          Swig_mark_arg(argIndex);
          verboseBuildLevel = NewString(argv[argIndex + 1]);
          Swig_mark_arg(argIndex + 1);
        } else if (strcmp(argv[argIndex], "-builderflagscript") == 0) {
          Swig_mark_arg(argIndex);
          buildFlagsScript = NewString(argv[argIndex + 1]);
          Swig_mark_arg(argIndex + 1);
        } else if (strcmp(argv[argIndex], "-gatewayxml") == 0) {
          Swig_mark_arg(argIndex);
          createGatewayXML = true;
          gatewayID = NewString(argv[argIndex + 1]);
          Swig_mark_arg(argIndex + 1);
        } else if (strcmp(argv[argIndex], "-gatewayxml6") == 0) {
          Swig_mark_arg(argIndex);
          createGatewayXMLV6 = true;
        }
      }
    }

    if (verboseBuildLevel == NULL) {
      verboseBuildLevel = NewString("0");
    }

    /* Set language-specific subdirectory in SWIG library */
    SWIG_library_directory("scilab");

    /* Add a symbol to the parser for conditional compilation */
    Preprocessor_define("SWIGSCILAB 1", 0);

    /* Set scilab configuration file */
    SWIG_config_file("scilab.swg");

    allow_overloading();
  }

  /* ------------------------------------------------------------------------
   * top()
   * ----------------------------------------------------------------------*/

  virtual int top(Node *node) {

    /* Get the module name */
    String *gatewayName = Getattr(node, "name");

    // Set library name
    String *gatewayLibraryName = NewStringf("lib%s", gatewayName);

    /* Get the output file name */
    String *outputFilename = Getattr(node, "outfile");

    /* Initialize I/O */
    beginSection = NewFile(outputFilename, "w", SWIG_output_files());
    if (!beginSection) {
      FileErrorDisplay(outputFilename);
      Exit(EXIT_FAILURE);
    }
    runtimeSection = NewString("");
    initSection = NewString("");
    headerSection = NewString("");
    wrappersSection = NewString("");

    /* Register file targets with the SWIG file handler */
    Swig_register_filebyname("begin", beginSection);
    Swig_register_filebyname("header", headerSection);
    Swig_register_filebyname("wrapper", wrappersSection);
    Swig_register_filebyname("runtime", runtimeSection);
    Swig_register_filebyname("init", initSection);

    /* Output module initialization code */
    Swig_banner(beginSection);

    Swig_obligatory_macros(runtimeSection, "SCILAB");

    // Gateway header source merged with wrapper source in nobuilder mode
    if (!generateBuilder)
      startGatewayHeader(gatewayLibraryName);

    // Create builder file if required
    if (generateBuilder) {
      createBuilderFile(outputFilename);
    }

    // Create gateway XML if required
    if (createGatewayXML) {
      createGatewayXMLFile(gatewayName);
    }

    // Create gateway XML V6 if required
    if (createGatewayXMLV6) {
      createGatewayXMLFileV6(gatewayName);
    }

    // Create loader script if required
    if (createLoader) {
      createLoaderFile(gatewayLibraryName);
    }

    // Module initialization function
    String *smallFunctionName = createSmallIdentifierName(gatewayName, SCILAB_IDENTIFIER_NAME_CHAR_MAX - 5);
    String *gatewayInitFunctionName = NewStringf("%s_Init", gatewayName);
    String *gatewayInitSmallFunctionName = NewStringf("%s_Init", smallFunctionName);
    String *wrapperFunctionName = NewStringf("SWIG_%s_Init", gatewayName);

    /* Add initialization function to builder table */
    addFunctionToScilab(gatewayInitFunctionName, gatewayInitSmallFunctionName, wrapperFunctionName);

    // Add helper functions to builder table
    addHelperFunctions();

    // Open Scilab wrapper variables creation function
    variablesCode = NewString("");
    Printf(variablesCode, "int SWIG_CreateScilabVariables(void *_pvApiCtx) {");

    /* Emit code for children */
    if (CPlusPlus) {
      Printf(wrappersSection, "extern \"C\" {\n");
    }

    Language::top(node);

    if (CPlusPlus) {
      Printf(wrappersSection, "}\n");
    }
    // Close Scilab wrapper variables creation function
    Printf(variablesCode, "  return SWIG_OK;\n}\n");

    // Add Builder footer code and save
    if (generateBuilder) {
      saveBuilderFile(gatewayLibraryName);
    }

    /* Close the init function and rename with module name */
    Printf(initSection, "return 0;\n}\n");
    Replaceall(initSection, "<module>", gatewayName);

    /* Write all to the wrapper file */
    SwigType_emit_type_table(runtimeSection, wrappersSection);	// Declare pointer types, ... (Ex: SWIGTYPE_p_p_double)

    // Gateway header source merged with wrapper source in nobuilder mode
    if (!generateBuilder) {
      terminateGatewayHeader(gatewayLibraryName);
      Printv(initSection, gatewayHeader, NIL);
    }

    Dump(runtimeSection, beginSection);
    Dump(headerSection, beginSection);
    Dump(wrappersSection, beginSection);
    Dump(variablesCode, beginSection);
    Wrapper_pretty_print(initSection, beginSection);

    if (createGatewayXML) {
      saveGatewayXMLFile();
    }
    if (createGatewayXMLV6) {
      saveGatewayXMLFileV6();
    }

    if (createLoader) {
      saveLoaderFile(gatewayLibraryName);
    }

    /* Cleanup files */
    Delete(runtimeSection);
    Delete(headerSection);
    Delete(wrappersSection);
    Delete(initSection);
    Delete(beginSection);

    Delete(sourceFileList);
    Delete(cflags);
    Delete(ldflags);

    return SWIG_OK;
  }

  /* ------------------------------------------------------------------------
   * emitBanner()
   * ----------------------------------------------------------------------*/

  void emitBanner(File *f) {
    Printf(f, "// ----------------------------------------------------------------------------\n");
    Swig_banner_target_lang(f, "// ");
    Printf(f, "// ----------------------------------------------------------------------------- */\n\n");
  }

  /* ------------------------------------------------------------------------
   * functionWrapper()
   * ----------------------------------------------------------------------*/

  virtual int functionWrapper(Node *node) {

    /* Get some useful attributes of this function */
    String *functionName = Getattr(node, "sym:name");
    String *smallFunctionName = createSmallIdentifierName(functionName);

    SwigType *functionReturnType = Getattr(node, "type");
    ParmList *functionParamsList = Getattr(node, "parms");

    int paramIndex = 0;		// Used for loops over ParmsList
    Parm *param = NULL;		// Used for loops over ParamsList

    /* Create the wrapper object */
    Wrapper *wrapper = NewWrapper();

    /* Create the function wrapper name */
    String *wrapperName = Swig_name_wrapper(functionName);

    /* Deal with overloading */
    String *overloadedName = Copy(wrapperName);
    /* Determine whether the function is overloaded or not */
    bool isOverloaded = ! !Getattr(node, "sym:overloaded");
    /* Determine whether the function is the last overloaded */
    bool isLastOverloaded = isOverloaded && !Getattr(node, "sym:nextSibling");

    if (!isOverloaded && !addSymbol(functionName, node)) {
      DelWrapper(wrapper);
      return SWIG_ERROR;
    }

    if (isOverloaded) {
      Append(overloadedName, Getattr(node, "sym:overname"));
    }

    /* Write the wrapper function definition (standard Scilab gateway function prototype) */
    Printv(wrapper->def, "SWIGEXPORT int ", overloadedName, "(SWIG_GatewayParameters) {", NIL);

    /* Emit all of the local variables for holding arguments */
    // E.g.: double arg1;
    emit_parameter_variables(functionParamsList, wrapper);

    /* Attach typemaps to the parameter list */
    // Add local variables used in typemaps (iRows, iCols, ...)
    emit_attach_parmmaps(functionParamsList, wrapper);
    Setattr(node, "wrap:parms", functionParamsList);

    /* Check input/output arguments count */
    int maxInputArguments = emit_num_arguments(functionParamsList);
    int minInputArguments = emit_num_required(functionParamsList);
    int minOutputArguments = 0;
    int maxOutputArguments = 0;

    if (!emit_isvarargs(functionParamsList)) {
      Printf(wrapper->code, "SWIG_CheckInputArgument(pvApiCtx, $mininputarguments, $maxinputarguments);\n");
    }
    else {
      Printf(wrapper->code, "SWIG_CheckInputArgumentAtLeast(pvApiCtx, $mininputarguments-1);\n");
    }
    Printf(wrapper->code, "SWIG_CheckOutputArgument(pvApiCtx, $minoutputarguments, $maxoutputarguments);\n");

    /* Set context */
    Printf(wrapper->code, "SWIG_Scilab_SetFuncName(fname);\n");
    Printf(wrapper->code, "SWIG_Scilab_SetApiContext(pvApiCtx);\n");

    /* Write typemaps(in) */

    for (paramIndex = 0, param = functionParamsList; paramIndex < maxInputArguments; ++paramIndex) {
      // Ignore parameter if the typemap specifies numinputs=0
      while (checkAttribute(param, "tmap:in:numinputs", "0")) {
	param = Getattr(param, "tmap:in:next");
      }

      SwigType *paramType = Getattr(param, "type");
      String *paramTypemap = Getattr(param, "tmap:in");

      if (paramTypemap) {
	// Replace $input by the position on Scilab stack
	String *source = NewString("");
	Printf(source, "%d", paramIndex + 1);
	Setattr(param, "emit:input", source);
	Replaceall(paramTypemap, "$input", Getattr(param, "emit:input"));

	if (Getattr(param, "wrap:disown") || (Getattr(param, "tmap:in:disown"))) {
	  Replaceall(paramTypemap, "$disown", "SWIG_POINTER_DISOWN");
	} else {
	  Replaceall(paramTypemap, "$disown", "0");
	}

	if (paramIndex >= minInputArguments) {	/* Optional input argument management */
	  Printf(wrapper->code, "if (SWIG_NbInputArgument(pvApiCtx) > %d) {\n%s\n}\n", paramIndex, paramTypemap);
	} else {
	  Printf(wrapper->code, "%s\n", paramTypemap);
	}
	param = Getattr(param, "tmap:in:next");
      } else {
	Swig_warning(WARN_TYPEMAP_IN_UNDEF, input_file, line_number, "Unable to use type %s as a function argument.\n", SwigType_str(paramType, 0));
	break;
      }
    }

    /* TODO write constraints */

    Setattr(node, "wrap:name", overloadedName);

    /* Emit the function call */
    Swig_director_emit_dynamic_cast(node, wrapper);
    String *functionActionCode = emit_action(node);

    /* Insert the return variable */
    emit_return_variable(node, functionReturnType, wrapper);

    /* Return the function value if necessary */
    String *functionReturnTypemap = Swig_typemap_lookup_out("out", node, Swig_cresult_name(), wrapper, functionActionCode);
    if (functionReturnTypemap) {
      // Result is actually the position of output value on stack
      if (Len(functionReturnTypemap) > 0) {
	Printf(wrapper->code, "SWIG_Scilab_SetOutputPosition(%d);\n", 1);
      }
      Replaceall(functionReturnTypemap, "$result", "1");

      if (GetFlag(node, "feature:new")) {
	Replaceall(functionReturnTypemap, "$owner", "1");
      } else {
	Replaceall(functionReturnTypemap, "$owner", "0");
      }

      Printf(wrapper->code, "%s\n", functionReturnTypemap);

      /* If the typemap is not empty, the function return one more argument than the typemaps gives */
      if (Len(functionReturnTypemap) > 0) {
	minOutputArguments++;
	maxOutputArguments++;
      }
      Delete(functionReturnTypemap);

    } else {
      Swig_warning(WARN_TYPEMAP_OUT_UNDEF, input_file, line_number, "Unable to use return type %s in function %s.\n", SwigType_str(functionReturnType, 0),
		   functionName);
    }

    /* Write typemaps(out) */
    for (param = functionParamsList; param;) {
      String *paramTypemap = Getattr(param, "tmap:argout");
      if (paramTypemap) {
	minOutputArguments++;
	maxOutputArguments++;
	Printf(wrapper->code, "SWIG_Scilab_SetOutputPosition(%d);\n", minOutputArguments);
	String *result = NewString("");
	Printf(result, "%d", minOutputArguments);
	Replaceall(paramTypemap, "$result", result);
	Printf(wrapper->code, "%s\n", paramTypemap);
	Delete(paramTypemap);
	param = Getattr(param, "tmap:argout:next");
      } else {
	param = nextSibling(param);
      }
    }
    /* Add cleanup code */
    for (param = functionParamsList; param;) {
      String *tm;
      if ((tm = Getattr(param, "tmap:freearg"))) {
	if (tm && (Len(tm) != 0)) {
	  Printf(wrapper->code, "%s\n", tm);
	}
	param = Getattr(param, "tmap:freearg:next");
      } else {
	param = nextSibling(param);
      }
    }

    /* See if there is any return cleanup code */
    String *tm;
    if ((tm = Swig_typemap_lookup("ret", node, Swig_cresult_name(), 0))) {
      Printf(wrapper->code, "%s\n", tm);
      Delete(tm);
    }

    /* Close the function(ok) */
    Printv(wrapper->code, "return SWIG_OK;\n", NIL);
    Printv(wrapper->code, "}\n", NIL);

    /* Add the failure cleanup code */
    /* TODO */

    /* Final substitutions if applicable */
    bool isvoid = !Cmp(functionReturnType, "void");
    Replaceall(wrapper->code, "$isvoid", isvoid ? "1" : "0");

    Replaceall(wrapper->code, "$symname", functionName);

    /* Set CheckInputArgument and CheckOutputArgument input arguments */
    if (maxOutputArguments < 1) {
      maxOutputArguments = 1;
    }
    if (minOutputArguments == 1) {
      minOutputArguments = 0;
    }
    String *argnumber = NewString("");
    Printf(argnumber, "%d", minInputArguments);
    Replaceall(wrapper->code, "$mininputarguments", argnumber);

    argnumber = NewString("");
    Printf(argnumber, "%d", maxInputArguments);
    Replaceall(wrapper->code, "$maxinputarguments", argnumber);

    argnumber = NewString("");
    Printf(argnumber, "%d", minOutputArguments);
    Replaceall(wrapper->code, "$minoutputarguments", argnumber);

    argnumber = NewString("");
    Printf(argnumber, "%d", maxOutputArguments);
    Replaceall(wrapper->code, "$maxoutputarguments", argnumber);

    /* Dump the function out */
    Wrapper_print(wrapper, wrappersSection);

    /* Update builder.sce contents */
    if (isLastOverloaded) {
      addFunctionToScilab(functionName, smallFunctionName, wrapperName);
      dispatchFunction(node);
    }

    if (!isOverloaded) {
      addFunctionToScilab(functionName, smallFunctionName, wrapperName);
    }

    /* tidy up */
    Delete(overloadedName);
    Delete(wrapperName);
    DelWrapper(wrapper);

    return SWIG_OK;
  }

  /* -----------------------------------------------------------------------
   * dispatchFunction()
   * ----------------------------------------------------------------------- */

  void dispatchFunction(Node *node) {
    Wrapper *wrapper = NewWrapper();

    String *functionName = Getattr(node, "sym:name");
    String *wrapperName = Swig_name_wrapper(functionName);
    int maxargs = 0;

    /* Generate the dispatch function */
    String *dispatch = Swig_overload_dispatch(node, "return %s(SWIG_GatewayArguments);", &maxargs);
    String *tmp = NewString("");

    Printv(wrapper->def, "SWIGEXPORT int ", wrapperName, "(SWIG_GatewayParameters) {\n", NIL);

    /* Get the number of the parameters */
    Wrapper_add_local(wrapper, "argc", "int argc = SWIG_NbInputArgument(pvApiCtx)");
    Printf(tmp, "int argv[%d] = {", maxargs);
    for (int j = 0; j < maxargs; ++j) {
      Printf(tmp, "%s%d", j ? "," : " ", j + 1);
    }
    Printf(tmp, "}");
    Wrapper_add_local(wrapper, "argv", tmp);

    Printf(wrapper->code, "SWIG_Scilab_SetApiContext(pvApiCtx);\n");

    /* Dump the dispatch function */
    Printv(wrapper->code, dispatch, "\n", NIL);
    Printf(wrapper->code, "Scierror(999, _(\"No matching function for overload\"));\n");
    Printf(wrapper->code, "return SWIG_ERROR;\n");
    Printv(wrapper->code, "}\n", NIL);
    Wrapper_print(wrapper, wrappersSection);

    Delete(tmp);
    DelWrapper(wrapper);
    Delete(dispatch);
    Delete(wrapperName);
  }

  /* -----------------------------------------------------------------------
   * variableWrapper()
   * ----------------------------------------------------------------------- */

  virtual int variableWrapper(Node *node) {

    /* Get information about variable */
    String *origVariableName = Getattr(node, "name");	// Ex: Shape::nshapes
    String *variableName = Getattr(node, "sym:name");	// Ex; Shape_nshapes (can be used for function names, ...)
    String *smallVariableName = createSmallIdentifierName(variableName, SCILAB_IDENTIFIER_NAME_CHAR_MAX - 4);

    /* Manage GET function */
    Wrapper *getFunctionWrapper = NewWrapper();
    String *getFunctionName = Swig_name_get(NSPACE_TODO, variableName);
    String *scilabGetFunctionName = Swig_name_get(NSPACE_TODO, variableName);
    String *scilabGetSmallFunctionName = Swig_name_get(NSPACE_TODO, smallVariableName);

    Setattr(node, "wrap:name", getFunctionName);
    Printv(getFunctionWrapper->def, "SWIGEXPORT int ", getFunctionName, "(SWIG_GatewayParameters) {\n", NIL);

    /* Check the number of input and output */
    Printf(getFunctionWrapper->def, "SWIG_CheckInputArgument(pvApiCtx, 0, 0);\n");
    Printf(getFunctionWrapper->def, "SWIG_CheckOutputArgument(pvApiCtx, 0, 1);\n");
    Printf(getFunctionWrapper->def, "SWIG_Scilab_SetApiContext(pvApiCtx);\n");

    String *varoutTypemap = Swig_typemap_lookup("varout", node, origVariableName, 0);
    if (varoutTypemap != NULL) {
      Printf(getFunctionWrapper->code, "SWIG_Scilab_SetOutputPosition(%d);\n", 1);
      Replaceall(varoutTypemap, "$value", origVariableName);
      Replaceall(varoutTypemap, "$result", "1");
      emit_action_code(node, getFunctionWrapper->code, varoutTypemap);
      Delete(varoutTypemap);
    }
    Append(getFunctionWrapper->code, "return SWIG_OK;\n");
    Append(getFunctionWrapper->code, "}\n");
    Wrapper_print(getFunctionWrapper, wrappersSection);

    /* Add function to builder table */
    addFunctionToScilab(scilabGetFunctionName, scilabGetSmallFunctionName, getFunctionName);

    /* Manage SET function */
    if (!is_immutable(node)) {
      Wrapper *setFunctionWrapper = NewWrapper();
      String *setFunctionName = Swig_name_set(NSPACE_TODO, variableName);
      String *scilabSetFunctionName = Swig_name_set(NSPACE_TODO, variableName);
      String *scilabSetSmallFunctionName = Swig_name_set(NSPACE_TODO, smallVariableName);

      Setattr(node, "wrap:name", setFunctionName);
      Printv(setFunctionWrapper->def, "SWIGEXPORT int ", setFunctionName, "(SWIG_GatewayParameters) {\n", NIL);

      /* Check the number of input and output */
      Printf(setFunctionWrapper->def, "SWIG_CheckInputArgument(pvApiCtx, 1, 1);\n");
      Printf(setFunctionWrapper->def, "SWIG_CheckOutputArgument(pvApiCtx, 0, 1);\n");
      Printf(setFunctionWrapper->def, "SWIG_Scilab_SetApiContext(pvApiCtx);\n");

      String *varinTypemap = Swig_typemap_lookup("varin", node, origVariableName, 0);
      if (varinTypemap != NULL) {
	Replaceall(varinTypemap, "$input", "1");
	emit_action_code(node, setFunctionWrapper->code, varinTypemap);
	Delete(varinTypemap);
      }
      Append(setFunctionWrapper->code, "return SWIG_OK;\n");
      Append(setFunctionWrapper->code, "}\n");
      Wrapper_print(setFunctionWrapper, wrappersSection);

      /* Add function to builder table */
      addFunctionToScilab(scilabSetFunctionName, scilabSetSmallFunctionName, setFunctionName);

      DelWrapper(setFunctionWrapper);
    }
    DelWrapper(getFunctionWrapper);

    return SWIG_OK;
  }

  /* -----------------------------------------------------------------------
   * constantWrapper()
   * ----------------------------------------------------------------------- */

  virtual int constantWrapper(Node *node) {

    /* Get the useful information from the node */
    String *nodeName = Getattr(node, "name");
    SwigType *type = Getattr(node, "type");
    String *constantName = Getattr(node, "sym:name");
    String *constantValue = Getattr(node, "value");
    String *constantTypemap = NULL;

    // If feature scilab:const enabled, constants & enums are wrapped to Scilab variables
    if (GetFlag(node, "feature:scilab:const")) {
      bool isConstant = ((SwigType_issimple(type)) || (SwigType_type(type) == T_STRING));
      bool isEnum = (Cmp(nodeType(node), "enumitem") == 0);

      if (isConstant || isEnum) {
	if (isEnum) {
	  Setattr(node, "type", "double");
	  constantValue = Getattr(node, "value");
	}

	constantTypemap = Swig_typemap_lookup("scilabconstcode", node, nodeName, 0);
	if (constantTypemap != NULL) {

	  Setattr(node, "wrap:name", constantName);
	  Replaceall(constantTypemap, "$result", constantName);
	  Replaceall(constantTypemap, "$value", constantValue);

	  emit_action_code(node, variablesCode, constantTypemap);
	  Delete(constantTypemap);
	  return SWIG_OK;
	}
      }
    }

    /* Create variables for member pointer constants, not supported by typemaps (like Python wrapper does) */
    if (SwigType_type(type) == T_MPOINTER) {
      String *wname = Swig_name_wrapper(constantName);
      String *str = SwigType_str(type, wname);
      Printf(headerSection, "static %s = %s;\n", str, constantValue);
      Delete(str);
      constantValue = wname;
    }

    // Constant names can have SCILAB_VARIABLE_NAME_CHAR_MAX because of suffixes "_get" added to function
    String *smallConstantName = createSmallIdentifierName(constantName, SCILAB_IDENTIFIER_NAME_CHAR_MAX - 4);

    /* Create GET function to get the constant value */
    Wrapper *getFunctionWrapper = NewWrapper();
    String *getFunctionName = Swig_name_get(NSPACE_TODO, constantName);
    String *scilabGetSmallFunctionName = Swig_name_get(NSPACE_TODO, smallConstantName);
    Setattr(node, "wrap:name", getFunctionName);
    Setattr(node, "wrap:name", getFunctionName);
    Printv(getFunctionWrapper->def, "SWIGEXPORT int ", getFunctionName, "(SWIG_GatewayParameters) {\n", NIL);

    /* Check the number of input and output */
    Printf(getFunctionWrapper->def, "SWIG_CheckInputArgument(pvApiCtx, 0, 0);\n");
    Printf(getFunctionWrapper->def, "SWIG_CheckOutputArgument(pvApiCtx, 0, 1);\n");
    Printf(getFunctionWrapper->def, "SWIG_Scilab_SetApiContext(pvApiCtx);\n");

    constantTypemap = Swig_typemap_lookup("constcode", node, nodeName, 0);
    if (constantTypemap != NULL) {
      Printf(getFunctionWrapper->code, "SWIG_Scilab_SetOutputPosition(%d);\n", 1);
      Replaceall(constantTypemap, "$value", constantValue);
      Replaceall(constantTypemap, "$result", "1");
      emit_action_code(node, getFunctionWrapper->code, constantTypemap);
      Delete(constantTypemap);
    }

    /* Dump the wrapper function */
    Append(getFunctionWrapper->code, "return SWIG_OK;\n");
    Append(getFunctionWrapper->code, "}\n");
    Wrapper_print(getFunctionWrapper, wrappersSection);

    /* Add the function to Scilab  */
    addFunctionToScilab(getFunctionName, scilabGetSmallFunctionName, getFunctionName);

    DelWrapper(getFunctionWrapper);

    return SWIG_OK;
  }

  /* ---------------------------------------------------------------------
   * enumvalueDeclaration()
   * --------------------------------------------------------------------- */

  virtual int enumvalueDeclaration(Node *node) {
    static int iPreviousEnumValue = 0;

    if (GetFlag(node, "feature:scilab:const")) {
      // Compute the "absolute" value of enum if needed
      // (most of time enum values are a linked list of relative values)
      String *enumValue = Getattr(node, "enumvalue");
      String *enumValueEx = Getattr(node, "enumvalueex");

      // First enum value ?
      String *firstenumitem = Getattr(node, "firstenumitem");
      if (firstenumitem) {
	if (enumValue) {
	  // Value is in 'enumvalue'
	  iPreviousEnumValue = atoi(Char(enumValue));
	} else if (enumValueEx) {
	  // Or value is in 'enumValueEx'
	  iPreviousEnumValue = atoi(Char(enumValueEx));

	  enumValue = NewString("");
	  Printf(enumValue, "%d", iPreviousEnumValue);
	  Setattr(node, "enumvalue", enumValue);
	}
      } else if (!enumValue && enumValueEx) {
	// Value is not specified, set it by incrementing last value
	enumValue = NewString("");
	Printf(enumValue, "%d", ++iPreviousEnumValue);
	Setattr(node, "enumvalue", enumValue);
      }
      // Enums in Scilab are mapped to double
      Setattr(node, "type", "double");
    }

    return Language::enumvalueDeclaration(node);
  }

  /* -----------------------------------------------------------------------
   * addHelperFunctions()
   * ----------------------------------------------------------------------- */

  void addHelperFunctions() {
    addFunctionToScilab("SWIG_this", "SWIG_this", "SWIG_this");
    addFunctionToScilab("SWIG_ptr", "SWIG_ptr", "SWIG_ptr");
  }

  /* -----------------------------------------------------------------------
   * addFunctionToScilab()
   * Declare a wrapped function in Scilab (builder, gateway, XML, ...)
   * ----------------------------------------------------------------------- */

  void addFunctionToScilab(const_String_or_char_ptr scilabFunctionName, const_String_or_char_ptr scilabSmallFunctionName, const_String_or_char_ptr wrapperFunctionName) {
    if (!generateBuilder)
      addFunctionInGatewayHeader(scilabFunctionName, scilabSmallFunctionName, wrapperFunctionName);

    if (generateBuilder) {
      addFunctionInScriptTable(scilabFunctionName, scilabSmallFunctionName, wrapperFunctionName, builderCode5, builderCode6);
    }

    if (createLoader) {
      addFunctionInLoader(scilabFunctionName, scilabSmallFunctionName);
    }

    if (gatewayXMLFile) {
      Printf(gatewayXML, "<PRIMITIVE gatewayId=\"%s\" primitiveId=\"%d\" primitiveName=\"%s\"/>\n", gatewayID, primitiveID++, scilabSmallFunctionName);
    }

    if (gatewayXMLFileV6) {
      Printf(gatewayXMLV6, "<gateway name=\"%s\" function=\"%s\" type=\"0\"/>\n", scilabFunctionName, scilabFunctionName);
    }
  }


  /* -----------------------------------------------------------------------
   * createBuilderCode()
   * ----------------------------------------------------------------------- */

  void createBuilderFile(String *outputFilename) {
    String *builderFilename = NewStringf("builder.sce");
    builderFile = NewFile(builderFilename, "w", SWIG_output_files());
    if (!builderFile) {
      FileErrorDisplay(builderFilename);
      Exit(EXIT_FAILURE);
    }
    emitBanner(builderFile);

    builderFunctionCount = 0;
    builderCode = NewString("");
    builderCode5 = NewString("");
    builderCode6 = NewString("");
    Printf(builderCode, "mode(-1);\n");
    Printf(builderCode, "lines(0);\n");	/* Useful for automatic tests */

    // Scilab needs to be in the build directory
    Printf(builderCode, "originaldir = pwd();\n");
    Printf(builderCode, "builddir = get_absolute_file_path('builder.sce');\n");
    Printf(builderCode, "cd(builddir);\n");

    Printf(builderCode, "ilib_verbose(%s);\n", verboseBuildLevel);

    Printf(builderCode, "libs = [];\n");

    // Flags from command line arguments
    Printf(builderCode, "cflags = \"\";\n");
    for (int i = 0; i < Len(cflags); i++) {
      String *cflag = Getitem(cflags, i);
      Printf(builderCode, "cflags = cflags + \" %s\";\n", cflag);
    }

    if (Len(ldflags) > 0) {
      for (int i = 0; i < Len(ldflags); i++) {
	String *ldflag = Getitem(ldflags, i);
	if (i == 0) {
	  Printf(builderCode, "ldflags = \"%s\";\n", ldflag);
	} else {
	  Printf(builderCode, "ldflags = ldflags + \" %s\";\n", ldflag);
	}
      }
    } else {
      Printf(builderCode, "ldflags = \"\";\n");
    }

    // External script to set flags
    if (buildFlagsScript) {
      Printf(builderCode, "exec(\"%s\");\n", buildFlagsScript);
      Printf(builderCode, "cflags = cflags + getCompilationFlags();\n");
      Printf(builderCode, "ldflags = ldflags + getLinkFlags();\n");
    }
    // Additional sources
    Insert(sourceFileList, 0, outputFilename);
    for (int i = 0; i < Len(sourceFileList); i++) {
      String *sourceFile = Getitem(sourceFileList, i);
      if (i == 0) {
	Printf(builderCode, "files = \"%s\";\n", sourceFile);
      } else {
	Printf(builderCode, "files($ + 1) = \"%s\";\n", sourceFile);
      }
    }

    Printf(builderCode5, "table = [ ..\n");
    Printf(builderCode6, "table = [ ..\n");
  }

  /* -----------------------------------------------------------------------
   * addFunctionInBuilderCode()
   * Add a function wrapper in the function table of generated builder script
   * ----------------------------------------------------------------------- */

  void addFunctionInScriptTable(const_String_or_char_ptr scilabFunctionName, const_String_or_char_ptr scilabSmallFunctionName, const_String_or_char_ptr wrapperFunctionName, String *scriptCode5, String *scriptCode6) {
    if (++builderFunctionCount % 10 == 0) {
      Printf(scriptCode5, "];\ntable = [table; ..\n");
      Printf(scriptCode6, "];\ntable = [table; ..\n");
    }
    Printf(scriptCode5, "\"%s\",\"%s\"; ..\n", scilabSmallFunctionName, wrapperFunctionName);
    Printf(scriptCode6, "\"%s\",\"%s\"; ..\n", scilabFunctionName, wrapperFunctionName);
  }

  /* -----------------------------------------------------------------------
   * saveBuilderFile()
   * ----------------------------------------------------------------------- */

  void saveBuilderFile(String *gatewayName) {
    Printf(builderCode5, "];\n");
    Printf(builderCode6, "];\n");

    if (Equal(builderCode5, builderCode6)) {
      Append(builderCode, builderCode6);
    } else {
      Printf(builderCode, "ver = getversion('scilab');\n");
      Printf(builderCode, "if ver(1) < 6 then\n");
      Printf(builderCode, "  // version is less or equal to 5.5.2\n");
      Printf(builderCode, "  \n");
      Append(builderCode, builderCode5);
      Printf(builderCode, "  \n");
      Printf(builderCode, "else\n");
      Printf(builderCode, "  // version is 6.0.0 or more\n");
      Printf(builderCode, "  \n");
      Append(builderCode, builderCode6);
      Printf(builderCode, "  \n");
      Printf(builderCode, "end\n");
    }

    Printf(builderCode, "ierr = 0;\n");
    Printf(builderCode, "if ~isempty(table) then\n");
    Printf(builderCode, "  ierr = execstr(\"ilib_build(''%s'', table, files, libs, [], ldflags, cflags);\", 'errcatch');\n", gatewayName);
    Printf(builderCode, "  if ierr <> 0 then\n");
    Printf(builderCode, "    err_msg = lasterror();\n");
    Printf(builderCode, "  end\n");
    Printf(builderCode, "end\n");
    Printf(builderCode, "cd(originaldir);\n");
    Printf(builderCode, "if ierr <> 0 then\n");
    Printf(builderCode, "  error(ierr, err_msg);\n");
    Printf(builderCode, "end\n");
    Printv(builderFile, builderCode, NIL);

    Delete(builderCode);
    Delete(builderFile);
  }

  /* -----------------------------------------------------------------------
   * createGatewayXMLFileV6()
   * This XML file is used by Scilab 6 in the context of internal modules or
   * to get the function list.
   * ----------------------------------------------------------------------- */

  void createGatewayXMLFileV6(String *gatewayName) {
    String *gatewayXMLFilename = NewStringf("%s_gateway.xml", gatewayName);
    gatewayXMLFileV6 = NewFile(gatewayXMLFilename, "w", SWIG_output_files());
    if (!gatewayXMLFileV6) {
      FileErrorDisplay(gatewayXMLFilename);
      Exit(EXIT_FAILURE);
    }
    // Add a slightly modified SWIG banner to the gateway XML ("--modify" is illegal in XML)
    gatewayXMLV6 = NewString("");
    Printf(gatewayXMLV6, "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n");
    Printf(gatewayXMLV6, "<!DOCTYPE module SYSTEM \"../../functions/xml/gateway.dtd\">\n");
    Printf(gatewayXMLV6, "<!--\n");
    Swig_banner_target_lang(gatewayXMLV6, "");
    Printf(gatewayXMLV6, "-->\n");
    Printf(gatewayXMLV6, "<module name=\"%s\">\n", gatewayName);
  }

  /* -----------------------------------------------------------------------
   * saveGatewayXMLFileV6()
   * ----------------------------------------------------------------------- */

  void saveGatewayXMLFileV6() {
    Printf(gatewayXMLV6, "</module>\n");
    Printv(gatewayXMLFileV6, gatewayXMLV6, NIL);
    Delete(gatewayXMLFileV6);
  }

  /* -----------------------------------------------------------------------
   * createGatewayXMLFile()
   * This XML file is used by Scilab in the context of internal modules
   * ----------------------------------------------------------------------- */

  void createGatewayXMLFile(String *gatewayName) {
    String *gatewayXMLFilename = NewStringf("%s_gateway.xml", gatewayName);
    gatewayXMLFile = NewFile(gatewayXMLFilename, "w", SWIG_output_files());
    if (!gatewayXMLFile) {
      FileErrorDisplay(gatewayXMLFilename);
      Exit(EXIT_FAILURE);
    }
    // Add a slightly modified SWIG banner to the gateway XML ("--modify" is illegal in XML)
    gatewayXML = NewString("");
    Printf(gatewayXML, "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n");
    Printf(gatewayXML, "<!--\n");
    Swig_banner_target_lang(gatewayXML, "");
    Printf(gatewayXML, "-->\n");
    Printf(gatewayXML, "<GATEWAY name=\"%s\">\n", gatewayName);

    primitiveID = 1;
  }

  /* -----------------------------------------------------------------------
   * saveGatewayXMLFile()
   * ----------------------------------------------------------------------- */

  void saveGatewayXMLFile() {
    Printf(gatewayXML, "</GATEWAY>\n");
    Printv(gatewayXMLFile, gatewayXML, NIL);
    Delete(gatewayXMLFile);
  }

  /* -----------------------------------------------------------------------
   * startGatewayHeader()
   * Start the gateway header
   * ----------------------------------------------------------------------- */
  void startGatewayHeader(String *gatewayLibraryName) {
    gatewayHeader = NewString("");
    Printf(gatewayHeader, "\n");

    gatewayHeaderV6 = NewString("");
    Printf(gatewayHeaderV6, "#ifdef __cplusplus\n");
    Printf(gatewayHeaderV6, "extern \"C\" {\n");
    Printf(gatewayHeaderV6, "#endif\n");
    Printf(gatewayHeaderV6, "#include \"c_gateway_prototype.h\"\n");
    Printf(gatewayHeaderV6, "#include \"addfunction.h\"\n");
    Printf(gatewayHeaderV6, "#ifdef __cplusplus\n");
    Printf(gatewayHeaderV6, "}\n");
    Printf(gatewayHeaderV6, "#endif\n");
    Printf(gatewayHeaderV6, "\n");
    Printf(gatewayHeaderV6, "#define MODULE_NAME L\"%s\"\n", gatewayLibraryName);
    Printf(gatewayHeaderV6, "#ifdef __cplusplus\n");
    Printf(gatewayHeaderV6, "extern \"C\"\n");
    Printf(gatewayHeaderV6, "#endif\n");
    Printf(gatewayHeaderV6, "SWIGEXPORT int %s(wchar_t *pwstFuncName) {\n", gatewayLibraryName);
    Printf(gatewayHeaderV6, "\n");
  }

  /* -----------------------------------------------------------------------
   * addFunctionInGatewayHeader()
   * Add a function in the gateway header
   * ----------------------------------------------------------------------- */

  void addFunctionInGatewayHeader(const_String_or_char_ptr scilabFunctionName, const_String_or_char_ptr scilabSmallFunctionName, const_String_or_char_ptr wrapperFunctionName) {
    if (gatewayHeaderV5 == NULL) {
      gatewayHeaderV5 = NewString("");
      Printf(gatewayHeaderV5, "static GenericTable Tab[] = {\n");
    } else
      Printf(gatewayHeaderV5, ",\n");
    Printf(gatewayHeaderV5, " {(Myinterfun)sci_gateway, (GT)%s, (char *)\"%s\"}", wrapperFunctionName, scilabSmallFunctionName);

    Printf(gatewayHeaderV6, "if (wcscmp(pwstFuncName, L\"%s\") == 0) { addCStackFunction((wchar_t *)L\"%s\", &%s, (wchar_t *)MODULE_NAME); }\n", scilabFunctionName, scilabFunctionName, wrapperFunctionName);
  }

  /* -----------------------------------------------------------------------
   * terminateGatewayHeader()
   * Terminates the gateway header
   * ----------------------------------------------------------------------- */

  void terminateGatewayHeader(String *gatewayLibraryName) {
    Printf(gatewayHeaderV5, "};\n");
    Printf(gatewayHeaderV5, "\n");
    Printf(gatewayHeaderV5, "#ifdef __cplusplus\n");
    Printf(gatewayHeaderV5, "extern \"C\" {\n");
    Printf(gatewayHeaderV5, "#endif\n");
    Printf(gatewayHeaderV5, "SWIGEXPORT int C2F(%s)() {\n", gatewayLibraryName);
    Printf(gatewayHeaderV5, "  Rhs = Max(0, Rhs);\n");
    Printf(gatewayHeaderV5, "  if (*(Tab[Fin-1].f) != NULL) {\n");
    Printf(gatewayHeaderV5, "    if(pvApiCtx == NULL) {\n");
    Printf(gatewayHeaderV5, "      pvApiCtx = (StrCtx *)MALLOC(sizeof(StrCtx));\n");
    Printf(gatewayHeaderV5, "    }\n");
    Printf(gatewayHeaderV5, "    pvApiCtx->pstName = (char *)Tab[Fin-1].name;\n");
    Printf(gatewayHeaderV5, "    (*(Tab[Fin-1].f))(Tab[Fin-1].name,(GatefuncH)Tab[Fin-1].F);\n");
    Printf(gatewayHeaderV5, "  }\n");
    Printf(gatewayHeaderV5, "  return 0;\n");
    Printf(gatewayHeaderV5, "}\n");
    Printf(gatewayHeaderV5, "\n");
    Printf(gatewayHeaderV5, "#ifdef __cplusplus\n");
    Printf(gatewayHeaderV5, "}\n");
    Printf(gatewayHeaderV5, "#endif\n");

    Printf(gatewayHeaderV6, "return 1;\n");
    Printf(gatewayHeaderV6, "};\n");

    Printf(gatewayHeader, "#if SCI_VERSION_MAJOR < 6\n");
    Printv(gatewayHeader, gatewayHeaderV5, NIL);
    Printf(gatewayHeader, "#else\n");
    Printv(gatewayHeader, gatewayHeaderV6, NIL);
    Printf(gatewayHeader, "#endif\n");
  }


  /* -----------------------------------------------------------------------
   * createLoaderScriptFile()
   * Creates the loader script file (loader.sce)
   * ----------------------------------------------------------------------- */

  void createLoaderFile(String *gatewayLibraryName) {
    String *loaderFilename = NewString("loader.sce");
    loaderFile = NewFile(loaderFilename, "w", SWIG_output_files());
    if (!loaderFile) {
      FileErrorDisplay(loaderFilename);
      Exit(EXIT_FAILURE);
    }

    emitBanner(loaderFile);

    loaderFunctionCount = 0;
    loaderScript = NewString("function loader_function()\n");
    Printf(loaderScript, "  p = get_absolute_file_path('loader.sce');\n", gatewayLibraryName);
    Printf(loaderScript, "  [bOK, ilib] = c_link('%s');\n", gatewayLibraryName);
    Printf(loaderScript, "  if bOK then\n");
    Printf(loaderScript, "    ulink(ilib);\n");
    Printf(loaderScript, "  end\n");
    loaderScript5 = NewString("    list_functions = [ ..\n");
    loaderScript6 = NewString("    list_functions = [ ..\n");
  }

  /* -----------------------------------------------------------------------
   * addFunctionInLoaderScript()
   * Add a function in the loader script table
   * ----------------------------------------------------------------------- */

  void addFunctionInLoader(const_String_or_char_ptr scilabFunctionName, const_String_or_char_ptr scilabSmallFunctionName) {
    if (++loaderFunctionCount % 10 == 0) {
      Printf(loaderScript5, "    ];\n    list_functions = [list_functions; ..\n");
      Printf(loaderScript6, "    ];\n    list_functions = [list_functions; ..\n");
    }
    Printf(loaderScript5, "      '%s'; ..\n", scilabSmallFunctionName);
    Printf(loaderScript6, "      '%s'; ..\n", scilabFunctionName);
  }

  /* -----------------------------------------------------------------------
   * saveLoaderScriptFile()
   * Terminates and saves the loader script
   * ----------------------------------------------------------------------- */

  void saveLoaderFile(String *gatewayLibraryName) {
    Printf(loaderScript5, "    ];\n");
    Printf(loaderScript6, "    ];\n");

    if (Equal(loaderScript5, loaderScript6)) {
      Append(loaderScript, loaderScript6);
    } else {
      Printf(loaderScript, "  ver = getversion('scilab');\n");
      Printf(loaderScript, "  if ver(1) < 6 then\n");
      Printf(loaderScript, "    // version is less or equal to 5.5.2\n");
      Printf(loaderScript, "    \n");
      Append(loaderScript, loaderScript5);
      Delete(loaderScript5);
      Printf(loaderScript, "    \n");
      Printf(loaderScript, "  else\n");
      Printf(loaderScript, "    // version is 6.0.0 or more\n");
      Printf(loaderScript, "    \n");
      Append(loaderScript, loaderScript6);
      Delete(loaderScript6);
      Printf(loaderScript, "    \n");
      Printf(loaderScript, "  end\n");
    }

    Printf(loaderScript, "  addinter(p + '%s' + getdynlibext(), '%s', list_functions);\n", gatewayLibraryName, gatewayLibraryName);
    Printf(loaderScript, "endfunction\n");
    Printf(loaderScript, "loader_function();\n");
    Printf(loaderScript, "clear loader_function;\n");
    Printv(loaderFile, loaderScript, NIL);

    Delete(loaderScript);
    Delete(loaderFile);
  }

  /* -----------------------------------------------------------------------
   * createSmallIdentifierName()
   * Create a Scilab small identifier to be used by Scilab 5
   * ----------------------------------------------------------------------- */

  String* createSmallIdentifierName(String* name, int outputLen = SCILAB_IDENTIFIER_NAME_CHAR_MAX) {
    char* s = Char(name);
    int nameLen = Len(s);

    // truncate and preserve common suffix
    if (outputLen > 4 && nameLen > outputLen) {
      String* smallName = NewStringWithSize(name, outputLen);
      char* smallNameStr = (char*) Data(smallName);

      if (s[nameLen-4] == '_' && s[nameLen - 3] == 'g' && s[nameLen - 2] == 'e' && s[nameLen - 1] == 't') {
        // get
        memcpy(&smallNameStr[outputLen - 4], &s[nameLen - 4], 4);
      } else if (s[nameLen-4] == '_' && s[nameLen - 3] == 's' && s[nameLen - 2] == 'e' && s[nameLen - 1] == 't') {
        // set
        memcpy(&smallNameStr[outputLen - 4], &s[nameLen - 4], 4);
      }

      return smallName;
    }
    
    return name;
  }

};

extern "C" Language *swig_scilab(void) {
  return new SCILAB();
}
