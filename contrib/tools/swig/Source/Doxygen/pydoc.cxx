/* -----------------------------------------------------------------------------
 * This file is part of SWIG, which is licensed as a whole under version 3
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at https://www.swig.org/legal.html.
 *
 * pydoc.cxx
 *
 * Module to return documentation for nodes formatted for PyDoc
 * ----------------------------------------------------------------------------- */

#include "pydoc.h"
#include "doxyparser.h"
#include <sstream>
#include <string>
#include <vector>
#include <iostream>

#include "swigmod.h"

// define static tables, they are filled in PyDocConverter's constructor
PyDocConverter::TagHandlersMap PyDocConverter::tagHandlers;
std::map<std::string, std::string> PyDocConverter::sectionTitles;

using std::string;

// Helper class increasing the provided indent string in its ctor and decreasing
// it in its dtor.
class IndentGuard {
public:
  // One indent level.
  static const char *Level() {
    return "    ";
  }
  // Default ctor doesn't do anything and prevents the dtor from doing anything// too and should only be used when the guard needs to be initialized// conditionally as Init() can then be called after checking some condition.// Otherwise, prefer to use the non default ctor below.
      IndentGuard() {
    m_initialized = false;
  }

  // Ctor takes the output to determine the current indent and to remove the
  // extra indent added to it in the dtor and the variable containing the indent
  // to use, which must be used after every new line by the code actually
  // updating the output.
  IndentGuard(string &output, string &indent) {
    Init(output, indent);
  }

  // Really initializes the object created using the default ctor.
  void Init(string &output, string &indent) {
    m_output = &output;
    m_indent = &indent;

    const string::size_type lastNonSpace = m_output->find_last_not_of(' ');
    if (lastNonSpace == string::npos) {
      m_firstLineIndent = m_output->length();
    } else if ((*m_output)[lastNonSpace] == '\n') {
      m_firstLineIndent = m_output->length() - (lastNonSpace + 1);
    } else {
      m_firstLineIndent = 0;
    }

    // Notice that the indent doesn't include the first line indent because it's
    // implicit, i.e. it is present in the input and so is copied into the
    // output anyhow.
    *m_indent = Level();

    m_initialized = true;
  }

  // Get the indent for the first line of the paragraph, which is smaller than
  // the indent for the subsequent lines.
  string getFirstLineIndent() const {
    return string(m_firstLineIndent, ' ');
  }
 
  ~IndentGuard() {
    if (!m_initialized)
      return;

    m_indent->clear();

    // Get rid of possible remaining extra indent, e.g. if there were any trailing
    // new lines: we shouldn't add the extra indent level to whatever follows
    // this paragraph.
    static const size_t lenIndentLevel = strlen(Level());
    if (m_output->length() > lenIndentLevel) {
      const size_t start = m_output->length() - lenIndentLevel;
      if (m_output->compare(start, string::npos, Level()) == 0)
        m_output->erase(start);
    }
  }

private:
  string *m_output;
  string *m_indent;
  string::size_type m_firstLineIndent;
  bool m_initialized;

  IndentGuard(const IndentGuard &);
  IndentGuard &operator=(const IndentGuard &);
};

// Return the indent of the given multiline string, i.e. the maximal number of
// spaces present in the beginning of all its non-empty lines.
static size_t determineIndent(const string &s) {
  size_t minIndent = static_cast<size_t>(-1);

  for (size_t lineStart = 0; lineStart < s.length();) {
    const size_t lineEnd = s.find('\n', lineStart);
    const size_t firstNonSpace = s.find_first_not_of(' ', lineStart);

    // If inequality doesn't hold, it means that this line contains only spaces
    // (notice that this works whether lineEnd is valid or string::npos), in
    // which case it doesn't matter when determining the indent.
    if (firstNonSpace < lineEnd) {
      // Here we can be sure firstNonSpace != string::npos.
      const size_t lineIndent = firstNonSpace - lineStart;
      if (lineIndent < minIndent)
        minIndent = lineIndent;
    }

    if (lineEnd == string::npos)
      break;

    lineStart = lineEnd + 1;
  }

  return minIndent;
}

static void trimWhitespace(string &s) {
  const string::size_type lastNonSpace = s.find_last_not_of(' ');
  if (lastNonSpace == string::npos)
    s.clear();
  else
    s.erase(lastNonSpace + 1);
}

// Erase the first character in the string if it is a newline
static void eraseLeadingNewLine(string &s) {
  if (!s.empty() && s[0] == '\n')
    s.erase(s.begin());
}

// Erase the last character in the string if it is a newline
static void eraseTrailingNewLine(string &s) {
  if (!s.empty() && s[s.size() - 1] == '\n')
    s.erase(s.size() - 1);
}

// Check the generated docstring line by line and make sure that any
// code and verbatim blocks have an empty line preceding them, which
// is necessary for Sphinx.  Additionally, this strips any empty lines
// appearing at the beginning of the docstring.
static string padCodeAndVerbatimBlocks(const string &docString) {
  std::string result;

  std::istringstream iss(docString);

  // Initialize to false because there is no previous line yet
  bool lastLineWasNonBlank = false;
  
  for (string line; std::getline(iss, line); result += line) {
    if (!result.empty()) {
      // Terminate the previous line
      result += '\n';
    }

    const size_t pos = line.find_first_not_of(" \t");
    if (pos == string::npos) {
      lastLineWasNonBlank = false;
    } else {
      if (lastLineWasNonBlank &&
	  (line.compare(pos, 13, ".. code-block") == 0 ||
	   line.compare(pos, 7, ".. math") == 0 ||
	   line.compare(pos, 3, ">>>") == 0)) {
	// Must separate code or math blocks from the previous line
	result += '\n';
      }
      lastLineWasNonBlank = true;
    }
  }
  return result;
}

// Helper function to extract the option value from a command,
// e.g. param[in] -> in
static std::string getCommandOption(const std::string &command, char openChar, char closeChar) {
  string option;

  size_t opt_begin, opt_end;
  opt_begin = command.find(openChar);
  opt_end = command.find(closeChar);
  if (opt_begin != string::npos && opt_end != string::npos)
    option = command.substr(opt_begin+1, opt_end-opt_begin-1);

  return option;
}


/* static */
PyDocConverter::TagHandlersMap::mapped_type PyDocConverter::make_handler(tagHandler handler) {
  return make_pair(handler, std::string());
}

/* static */
PyDocConverter::TagHandlersMap::mapped_type PyDocConverter::make_handler(tagHandler handler, const char *arg) {
  return make_pair(handler, arg);
}

void PyDocConverter::fillStaticTables() {
  if (tagHandlers.size()) // fill only once
    return;

  // table of section titles, they are printed only once
  // for each group of specified doxygen commands
  sectionTitles["author"] = "Author: ";
  sectionTitles["authors"] = "Authors: ";
  sectionTitles["copyright"] = "Copyright: ";
  sectionTitles["deprecated"] = "Deprecated: ";
  sectionTitles["example"] = "Example: ";
  sectionTitles["note"] = "Notes: ";
  sectionTitles["remark"] = "Remarks: ";
  sectionTitles["remarks"] = "Remarks: ";
  sectionTitles["warning"] = "Warning: ";
//  sectionTitles["sa"] = "See also: ";
//  sectionTitles["see"] = "See also: ";
  sectionTitles["since"] = "Since: ";
  sectionTitles["todo"] = "TODO: ";
  sectionTitles["version"] = "Version: ";

  tagHandlers["a"] = make_handler(&PyDocConverter::handleTagWrap, "*");
  tagHandlers["b"] = make_handler(&PyDocConverter::handleTagWrap, "**");
  // \c command is translated as single quotes around next word
  tagHandlers["c"] = make_handler(&PyDocConverter::handleTagWrap, "``");
  tagHandlers["cite"] = make_handler(&PyDocConverter::handleTagWrap, "'");
  tagHandlers["e"] = make_handler(&PyDocConverter::handleTagWrap, "*");
  // these commands insert just a single char, some of them need to be escaped
  tagHandlers["$"] = make_handler(&PyDocConverter::handleTagChar);
  tagHandlers["@"] = make_handler(&PyDocConverter::handleTagChar);
  tagHandlers["\\"] = make_handler(&PyDocConverter::handleTagChar);
  tagHandlers["<"] = make_handler(&PyDocConverter::handleTagChar);
  tagHandlers[">"] = make_handler(&PyDocConverter::handleTagChar);
  tagHandlers["&"] = make_handler(&PyDocConverter::handleTagChar);
  tagHandlers["#"] = make_handler(&PyDocConverter::handleTagChar);
  tagHandlers["%"] = make_handler(&PyDocConverter::handleTagChar);
  tagHandlers["~"] = make_handler(&PyDocConverter::handleTagChar);
  tagHandlers["\""] = make_handler(&PyDocConverter::handleTagChar);
  tagHandlers["."] = make_handler(&PyDocConverter::handleTagChar);
  tagHandlers["::"] = make_handler(&PyDocConverter::handleTagChar);
  // these commands are stripped out, and only their content is printed
  tagHandlers["attention"] = make_handler(&PyDocConverter::handleParagraph);
  tagHandlers["author"] = make_handler(&PyDocConverter::handleParagraph);
  tagHandlers["authors"] = make_handler(&PyDocConverter::handleParagraph);
  tagHandlers["brief"] = make_handler(&PyDocConverter::handleParagraph);
  tagHandlers["bug"] = make_handler(&PyDocConverter::handleParagraph);
  tagHandlers["code"] = make_handler(&PyDocConverter::handleCode);
  tagHandlers["copyright"] = make_handler(&PyDocConverter::handleParagraph);
  tagHandlers["date"] = make_handler(&PyDocConverter::handleParagraph);
  tagHandlers["deprecated"] = make_handler(&PyDocConverter::handleParagraph);
  tagHandlers["details"] = make_handler(&PyDocConverter::handleParagraph);
  tagHandlers["em"] = make_handler(&PyDocConverter::handleTagWrap, "*");
  tagHandlers["example"] = make_handler(&PyDocConverter::handleParagraph);
  tagHandlers["exception"] = tagHandlers["throw"] = tagHandlers["throws"] = make_handler(&PyDocConverter::handleTagException);
  tagHandlers["htmlonly"] = make_handler(&PyDocConverter::handleParagraph);
  tagHandlers["invariant"] = make_handler(&PyDocConverter::handleParagraph);
  tagHandlers["latexonly"] = make_handler(&PyDocConverter::handleParagraph);
  tagHandlers["link"] = make_handler(&PyDocConverter::handleParagraph);
  tagHandlers["manonly"] = make_handler(&PyDocConverter::handleParagraph);
  tagHandlers["note"] = make_handler(&PyDocConverter::handleParagraph);
  tagHandlers["p"] = make_handler(&PyDocConverter::handleTagWrap, "``");
  tagHandlers["partofdescription"] = make_handler(&PyDocConverter::handleParagraph);
  tagHandlers["rtfonly"] = make_handler(&PyDocConverter::handleParagraph);
  tagHandlers["remark"] = make_handler(&PyDocConverter::handleParagraph);
  tagHandlers["remarks"] = make_handler(&PyDocConverter::handleParagraph);
  tagHandlers["sa"] = make_handler(&PyDocConverter::handleTagMessage, "See also: ");
  tagHandlers["see"] = make_handler(&PyDocConverter::handleTagMessage, "See also: ");
  tagHandlers["since"] = make_handler(&PyDocConverter::handleParagraph);
  tagHandlers["short"] = make_handler(&PyDocConverter::handleParagraph);
  tagHandlers["todo"] = make_handler(&PyDocConverter::handleParagraph);
  tagHandlers["version"] = make_handler(&PyDocConverter::handleParagraph);
  tagHandlers["verbatim"] = make_handler(&PyDocConverter::handleVerbatimBlock);
  tagHandlers["warning"] = make_handler(&PyDocConverter::handleParagraph);
  tagHandlers["xmlonly"] = make_handler(&PyDocConverter::handleParagraph);
  // these commands have special handlers
  tagHandlers["arg"] = make_handler(&PyDocConverter::handleTagMessage, "* ");
  tagHandlers["cond"] = make_handler(&PyDocConverter::handleTagMessage, "Conditional comment: ");
  tagHandlers["else"] = make_handler(&PyDocConverter::handleTagIf, "Else: ");
  tagHandlers["elseif"] = make_handler(&PyDocConverter::handleTagIf, "Else if: ");
  tagHandlers["endcond"] = make_handler(&PyDocConverter::handleTagMessage, "End of conditional comment.");
  tagHandlers["if"] = make_handler(&PyDocConverter::handleTagIf, "If: ");
  tagHandlers["ifnot"] = make_handler(&PyDocConverter::handleTagIf, "If not: ");
  tagHandlers["image"] = make_handler(&PyDocConverter::handleTagImage);
  tagHandlers["li"] = make_handler(&PyDocConverter::handleTagMessage, "* ");
  tagHandlers["overload"] = make_handler(&PyDocConverter::handleTagMessage,
                                         "This is an overloaded member function, provided for"
                                         " convenience.\nIt differs from the above function only in what" " argument(s) it accepts.");
  tagHandlers["par"] = make_handler(&PyDocConverter::handleTagPar);
  tagHandlers["param"] = tagHandlers["tparam"] = make_handler(&PyDocConverter::handleTagParam);
  tagHandlers["ref"] = make_handler(&PyDocConverter::handleTagRef);
  tagHandlers["result"] = tagHandlers["return"] = tagHandlers["returns"] = make_handler(&PyDocConverter::handleTagReturn);

  // this command just prints its contents
  // (it is internal command of swig's parser, contains plain text)
  tagHandlers["plainstd::string"] = make_handler(&PyDocConverter::handlePlainString);
  tagHandlers["plainstd::endl"] = make_handler(&PyDocConverter::handleNewLine);
  tagHandlers["n"] = make_handler(&PyDocConverter::handleNewLine);

  // \f commands output literal Latex formula, which is still better than nothing.
  tagHandlers["f$"] = tagHandlers["f["] = tagHandlers["f{"] = make_handler(&PyDocConverter::handleMath);

  // HTML tags
  tagHandlers["<a"] = make_handler(&PyDocConverter::handleDoxyHtmlTag_A);
  tagHandlers["<b"] = make_handler(&PyDocConverter::handleDoxyHtmlTag2, "**");
  tagHandlers["<blockquote"] = make_handler(&PyDocConverter::handleDoxyHtmlTag_A, "Quote: ");
  tagHandlers["<body"] = make_handler(&PyDocConverter::handleDoxyHtmlTag);
  tagHandlers["<br"] = make_handler(&PyDocConverter::handleDoxyHtmlTag, "\n");

  // there is no formatting for this tag as it was deprecated in HTML 4.01 and
  // not used in HTML 5
  tagHandlers["<center"] = make_handler(&PyDocConverter::handleDoxyHtmlTag);
  tagHandlers["<caption"] = make_handler(&PyDocConverter::handleDoxyHtmlTag);
  tagHandlers["<code"] = make_handler(&PyDocConverter::handleDoxyHtmlTag2, "``");

  tagHandlers["<dl"] = make_handler(&PyDocConverter::handleDoxyHtmlTag);
  tagHandlers["<dd"] = make_handler(&PyDocConverter::handleDoxyHtmlTag, "    ");
  tagHandlers["<dt"] = make_handler(&PyDocConverter::handleDoxyHtmlTag);

  tagHandlers["<dfn"] = make_handler(&PyDocConverter::handleDoxyHtmlTag);
  tagHandlers["<div"] = make_handler(&PyDocConverter::handleDoxyHtmlTag);
  tagHandlers["<em"] = make_handler(&PyDocConverter::handleDoxyHtmlTag2, "**");
  tagHandlers["<form"] = make_handler(&PyDocConverter::handleDoxyHtmlTag);
  tagHandlers["<hr"] = make_handler(&PyDocConverter::handleDoxyHtmlTag, "--------------------------------------------------------------------\n");
  tagHandlers["<h1"] = make_handler(&PyDocConverter::handleDoxyHtmlTag, "# ");
  tagHandlers["<h2"] = make_handler(&PyDocConverter::handleDoxyHtmlTag, "## ");
  tagHandlers["<h3"] = make_handler(&PyDocConverter::handleDoxyHtmlTag, "### ");
  tagHandlers["<i"] = make_handler(&PyDocConverter::handleDoxyHtmlTag2, "*");
  tagHandlers["<input"] = make_handler(&PyDocConverter::handleDoxyHtmlTag);
  tagHandlers["<img"] = make_handler(&PyDocConverter::handleDoxyHtmlTag, "Image:");
  tagHandlers["<li"] = make_handler(&PyDocConverter::handleDoxyHtmlTag, "* ");
  tagHandlers["<meta"] = make_handler(&PyDocConverter::handleDoxyHtmlTag);
  tagHandlers["<multicol"] = make_handler(&PyDocConverter::handleDoxyHtmlTag);
  tagHandlers["<ol"] = make_handler(&PyDocConverter::handleDoxyHtmlTag);
  tagHandlers["<p"] = make_handler(&PyDocConverter::handleDoxyHtmlTag, "\n");
  tagHandlers["<pre"] = make_handler(&PyDocConverter::handleDoxyHtmlTag);
  tagHandlers["<small"] = make_handler(&PyDocConverter::handleDoxyHtmlTag);
  tagHandlers["<span"] = make_handler(&PyDocConverter::handleDoxyHtmlTag2, "'");
  tagHandlers["<strong"] = make_handler(&PyDocConverter::handleDoxyHtmlTag2, "**");

  // make a space between text and super/sub script.
  tagHandlers["<sub"] = make_handler(&PyDocConverter::handleDoxyHtmlTag, " ");
  tagHandlers["<sup"] = make_handler(&PyDocConverter::handleDoxyHtmlTag, " ");

  tagHandlers["<table"] = make_handler(&PyDocConverter::handleDoxyHtmlTagNoParam);
  tagHandlers["<td"] = make_handler(&PyDocConverter::handleDoxyHtmlTag_td);
  tagHandlers["<th"] = make_handler(&PyDocConverter::handleDoxyHtmlTag_th);
  tagHandlers["<tr"] = make_handler(&PyDocConverter::handleDoxyHtmlTag_tr);
  tagHandlers["<tt"] = make_handler(&PyDocConverter::handleDoxyHtmlTag);
  tagHandlers["<kbd"] = make_handler(&PyDocConverter::handleDoxyHtmlTag);
  tagHandlers["<ul"] = make_handler(&PyDocConverter::handleDoxyHtmlTag);
  tagHandlers["<var"] = make_handler(&PyDocConverter::handleDoxyHtmlTag2, "*");

  // HTML entities
  tagHandlers["&copy"] = make_handler(&PyDocConverter::handleHtmlEntity, "(C)");
  tagHandlers["&trade"] = make_handler(&PyDocConverter::handleHtmlEntity, " TM");
  tagHandlers["&reg"] = make_handler(&PyDocConverter::handleHtmlEntity, "(R)");
  tagHandlers["&lt"] = make_handler(&PyDocConverter::handleHtmlEntity, "<");
  tagHandlers["&gt"] = make_handler(&PyDocConverter::handleHtmlEntity, ">");
  tagHandlers["&amp"] = make_handler(&PyDocConverter::handleHtmlEntity, "&");
  tagHandlers["&apos"] = make_handler(&PyDocConverter::handleHtmlEntity, "'");
  tagHandlers["&quot"] = make_handler(&PyDocConverter::handleHtmlEntity, "\"");
  tagHandlers["&lsquo"] = make_handler(&PyDocConverter::handleHtmlEntity, "`");
  tagHandlers["&rsquo"] = make_handler(&PyDocConverter::handleHtmlEntity, "'");
  tagHandlers["&ldquo"] = make_handler(&PyDocConverter::handleHtmlEntity, "\"");
  tagHandlers["&rdquo"] = make_handler(&PyDocConverter::handleHtmlEntity, "\"");
  tagHandlers["&ndash"] = make_handler(&PyDocConverter::handleHtmlEntity, "-");
  tagHandlers["&mdash"] = make_handler(&PyDocConverter::handleHtmlEntity, "--");
  tagHandlers["&nbsp"] = make_handler(&PyDocConverter::handleHtmlEntity, " ");
  tagHandlers["&times"] = make_handler(&PyDocConverter::handleHtmlEntity, "x");
  tagHandlers["&minus"] = make_handler(&PyDocConverter::handleHtmlEntity, "-");
  tagHandlers["&sdot"] = make_handler(&PyDocConverter::handleHtmlEntity, ".");
  tagHandlers["&sim"] = make_handler(&PyDocConverter::handleHtmlEntity, "~");
  tagHandlers["&le"] = make_handler(&PyDocConverter::handleHtmlEntity, "<=");
  tagHandlers["&ge"] = make_handler(&PyDocConverter::handleHtmlEntity, ">=");
  tagHandlers["&larr"] = make_handler(&PyDocConverter::handleHtmlEntity, "<--");
  tagHandlers["&rarr"] = make_handler(&PyDocConverter::handleHtmlEntity, "-->");
}

PyDocConverter::PyDocConverter(int flags):
DoxygenTranslator(flags), m_tableLineLen(0), m_prevRowIsTH(false) {
  fillStaticTables();
}

// Return the type as it should appear in the output documentation.
static std::string getPyDocType(Node *n, const_String_or_char_ptr lname = "") {
  std::string type;

  String *s = Swig_typemap_lookup("doctype", n, lname, 0);
  if (!s) {
    if (String *t = Getattr(n, "type"))
      s = SwigType_str(t, "");
  }

  if (!s)
    return type;

  if (Language::classLookup(s)) {
    // In Python C++ namespaces are flattened, so remove all but last component
    // of the name.
    String *const last = Swig_scopename_last(s);

    // We are not actually sure whether it's a documented class or not, but
    // there doesn't seem to be any harm in making it a reference if it isn't,
    // while there is a lot of benefit in having a hyperlink if it is.
    type = ":py:class:`";
    type += Char(last);
    type += "`";

    Delete(last);
  } else {
    type = Char(s);
  }

  Delete(s);

  return type;
}

std::string PyDocConverter::getParamType(std::string param) {
  std::string type;

  ParmList *plist = CopyParmList(Getattr(currentNode, "parms"));
  for (Parm *p = plist; p; p = nextSibling(p)) {
    String *pname = Getattr(p, "name");
    if (pname && Char(pname) == param) {
      type = getPyDocType(p, pname);
      break;
    }
  }
  Delete(plist);
  return type;
}

std::string PyDocConverter::getParamValue(std::string param) {
  std::string value;

  ParmList *plist = CopyParmList(Getattr(currentNode, "parms"));
  for (Parm *p = plist; p; p = nextSibling(p)) {
    String *pname = Getattr(p, "name");
    if (pname && Char(pname) == param) {
      String *pval = Getattr(p, "value");
      if (pval)
	value = Char(pval);
      break;
    }
  }
  Delete(plist);
  return value;
}

std::string PyDocConverter::translateSubtree(DoxygenEntity &doxygenEntity) {
  std::string translatedComment;

  if (doxygenEntity.isLeaf)
    return translatedComment;

  std::string currentSection;
  std::list<DoxygenEntity>::iterator p = doxygenEntity.entityList.begin();
  while (p != doxygenEntity.entityList.end()) {
    std::map<std::string, std::string>::iterator it;
    it = sectionTitles.find(p->typeOfEntity);
    if (it != sectionTitles.end()) {
      if (it->second != currentSection) {
        currentSection = it->second;
        translatedComment += currentSection;
      }
    }
    translateEntity(*p, translatedComment);
    translateSubtree(*p);
    p++;
  }

  return translatedComment;
}

void PyDocConverter::translateEntity(DoxygenEntity &doxyEntity, std::string &translatedComment) {
  // check if we have needed handler and call it
  std::map<std::string, std::pair<tagHandler, std::string> >::iterator it;
  it = tagHandlers.find(getBaseCommand(doxyEntity.typeOfEntity));
  if (it != tagHandlers.end())
    (this->*(it->second.first)) (doxyEntity, translatedComment, it->second.second);
}

void PyDocConverter::handleParagraph(DoxygenEntity &tag, std::string &translatedComment, const std::string &) {
  translatedComment += translateSubtree(tag);
}

void PyDocConverter::handleVerbatimBlock(DoxygenEntity &tag, std::string &translatedComment, const std::string &) {
  string verb = translateSubtree(tag);

  eraseLeadingNewLine(verb);

  // Remove the last newline to prevent doubling the newline already present after \endverbatim
  trimWhitespace(verb); // Needed to catch trailing newline below
  eraseTrailingNewLine(verb);
  translatedComment += verb;
}

void PyDocConverter::handleMath(DoxygenEntity &tag, std::string &translatedComment, const std::string &arg) {
  IndentGuard indent;

  // Only \f$ is translated to inline formulae, \f[ and \f{ are for the block ones.
  const bool inlineFormula = tag.typeOfEntity == "f$";

  string formulaNL;

  if (inlineFormula) {
    translatedComment += ":math:`";
  } else {
    indent.Init(translatedComment, m_indent);

    trimWhitespace(translatedComment);

    const string formulaIndent = indent.getFirstLineIndent();
    translatedComment += formulaIndent;
    translatedComment += ".. math::\n";

    formulaNL = '\n';
    formulaNL += formulaIndent;
    formulaNL += m_indent;
    translatedComment += formulaNL;
  }

  std::string formula;
  handleTagVerbatim(tag, formula, arg);

  // It is important to ensure that we have no spaces around the inline math
  // contents, so strip them.
  const size_t start = formula.find_first_not_of(" \t\n");
  const size_t end = formula.find_last_not_of(" \t\n");
  if (start != std::string::npos) {
    for (size_t n = start; n <= end; n++) {
      if (formula[n] == '\n') {
        // New lines must be suppressed in inline maths and indented in the block ones.
        if (!inlineFormula)
          translatedComment += formulaNL;
      } else {
        // Just copy everything else.
        translatedComment += formula[n];
      }
    }
  }

  if (inlineFormula) {
    translatedComment += "`";
  }
}

void PyDocConverter::handleCode(DoxygenEntity &tag, std::string &translatedComment, const std::string &arg) {
  IndentGuard indent(translatedComment, m_indent);

  trimWhitespace(translatedComment);

  // Check for an option given to the code command (e.g. code{.py}),
  // and try to set the code-block language accordingly.
  string option = getCommandOption(tag.typeOfEntity, '{', '}');
  // Set up the language option to the code-block command, which can
  // be any language supported by pygments:
  string codeLanguage;
  if (option == ".py")
    // Other possibilities here are "default" or "python3".  In Sphinx
    // 2.1.2, basic syntax doesn't render quite the same in these as
    // with "python", which for basic keywords seems to provide
    // slightly richer formatting.  Another option would be to leave
    // the language empty, but testing with Sphinx 1.8.5 has produced
    // an error "1 argument required".
    codeLanguage = "python";
  else if (option == ".java")
    codeLanguage = "java";
  else if (option == ".c")
    codeLanguage = "c";
  else
    // If there is not a match, or if no option was given, go out on a
    // limb and assume that the examples in the C or C++ sources use
    // C++.
    codeLanguage = "c++";

  std::string code;
  handleTagVerbatim(tag, code, arg);

  // Try and remove leading newline, which is present for block \code
  // command:
  eraseLeadingNewLine(code);

  // Check for python doctest blocks, and treat them specially:
  bool isDocTestBlock = false;
  size_t startPos;
  // ">>>" would normally appear at the beginning, but doxygen comment
  // style may have space in front, so skip leading whitespace
  if ((startPos=code.find_first_not_of(" \t")) != string::npos && code.substr(startPos,3) == ">>>")
    isDocTestBlock = true;

  string codeIndent;
  if (! isDocTestBlock) {
    // Use the current indent for the code-block line itself.
    translatedComment += indent.getFirstLineIndent();
    translatedComment += ".. code-block:: " + codeLanguage + "\n\n";

    // Specify the level of extra indentation that will be used for
    // subsequent lines within the code block.  Note that the correct
    // "starting indentation" is already present in the input, so we
    // only need to add the desired code block indentation.
    codeIndent = m_indent;
  }

  translatedComment += codeIndent;
  for (size_t n = 0; n < code.length(); n++) {
    if (code[n] == '\n') {
      // Don't leave trailing white space, this results in PEP8 validation
      // errors in Python code (which are performed by our own unit tests).
      trimWhitespace(translatedComment);
      translatedComment += '\n';

      // Ensure that we indent all the lines by the code indent.
      translatedComment += codeIndent;
    } else {
      // Just copy everything else.
      translatedComment += code[n];
    }
  }

  trimWhitespace(translatedComment);

  // For block commands, the translator adds the newline after
  // \endcode, so try and compensate by removing the last newline from
  // the code text:
  eraseTrailingNewLine(translatedComment);
}

void PyDocConverter::handlePlainString(DoxygenEntity &tag, std::string &translatedComment, const std::string &) {
  translatedComment += tag.data;
}

void PyDocConverter::handleTagVerbatim(DoxygenEntity &tag, std::string &translatedComment, const std::string &arg) {
  translatedComment += arg;
  for (DoxygenEntityListCIt it = tag.entityList.begin(); it != tag.entityList.end(); it++) {
    translatedComment += it->data;
  }
}

void PyDocConverter::handleTagMessage(DoxygenEntity &tag, std::string &translatedComment, const std::string &arg) {
  translatedComment += arg;
  handleParagraph(tag, translatedComment);
}

void PyDocConverter::handleTagChar(DoxygenEntity &tag, std::string &translatedComment, const std::string &) {
  translatedComment += tag.typeOfEntity;
}

void PyDocConverter::handleTagIf(DoxygenEntity &tag, std::string &translatedComment, const std::string &arg) {
  translatedComment += arg;
  if (tag.entityList.size()) {
    translatedComment += tag.entityList.begin()->data;
    tag.entityList.pop_front();
    translatedComment += " {" + translateSubtree(tag) + "}";
  }
}

void PyDocConverter::handleTagPar(DoxygenEntity &tag, std::string &translatedComment, const std::string &) {
  translatedComment += "Title: ";
  if (tag.entityList.size())
    translatedComment += tag.entityList.begin()->data;
  tag.entityList.pop_front();
  handleParagraph(tag, translatedComment);
}

void PyDocConverter::handleTagImage(DoxygenEntity &tag, std::string &translatedComment, const std::string &) {
  if (tag.entityList.size() < 2)
    return;
  tag.entityList.pop_front();
  translatedComment += "Image: ";
  translatedComment += tag.entityList.begin()->data;
  tag.entityList.pop_front();
  if (tag.entityList.size())
    translatedComment += "(" + tag.entityList.begin()->data + ")";
}

void PyDocConverter::handleTagParam(DoxygenEntity &tag, std::string &translatedComment, const std::string &) {
  if (tag.entityList.size() < 2)
    return;

  IndentGuard indent(translatedComment, m_indent);

  DoxygenEntity paramNameEntity = *tag.entityList.begin();
  tag.entityList.pop_front();

  const std::string &paramName = paramNameEntity.data;

  const std::string paramType = getParamType(paramName);
  const std::string paramValue = getParamValue(paramName);

  // Get command option, e.g. "in", "out", or "in,out"
  string commandOpt = getCommandOption(tag.typeOfEntity, '[', ']');
  if (commandOpt == "in,out") commandOpt = "in/out";

  // If provided, append the parameter direction to the type
  // information via a suffix:
  std::string suffix;
  if (commandOpt.size() > 0)
    suffix = ", " + commandOpt;
  
  // If the parameter has a default value, flag it as optional in the
  // generated type definition.  Particularly helpful when the python
  // call is generated with *args, **kwargs.
  if (paramValue.size() > 0)
    suffix += ", optional";  

  if (!paramType.empty()) {
    translatedComment += ":type " + paramName + ": " + paramType + suffix + "\n";
    translatedComment += indent.getFirstLineIndent();
  }

  translatedComment += ":param " + paramName + ":";

  handleParagraph(tag, translatedComment);
}


void PyDocConverter::handleTagReturn(DoxygenEntity &tag, std::string &translatedComment, const std::string &) {
  IndentGuard indent(translatedComment, m_indent);

  const std::string pytype = getPyDocType(currentNode);
  if (!pytype.empty()) {
    translatedComment += ":rtype: ";
    translatedComment += pytype;
    translatedComment += "\n";
    translatedComment += indent.getFirstLineIndent();
  }

  translatedComment += ":return: ";
  handleParagraph(tag, translatedComment);
}


void PyDocConverter::handleTagException(DoxygenEntity &tag, std::string &translatedComment, const std::string &) {
  IndentGuard indent(translatedComment, m_indent);

  translatedComment += ":raises: ";
  handleParagraph(tag, translatedComment);
}


void PyDocConverter::handleTagRef(DoxygenEntity &tag, std::string &translatedComment, const std::string &) {
  if (!tag.entityList.size())
    return;

  string anchor = tag.entityList.begin()->data;
  tag.entityList.pop_front();
  string anchorText = anchor;
  if (!tag.entityList.empty()) {
    anchorText = tag.entityList.begin()->data;
  }
  translatedComment += "'" + anchorText + "'";
}


void PyDocConverter::handleTagWrap(DoxygenEntity &tag, std::string &translatedComment, const std::string &arg) {
  if (tag.entityList.size()) { // do not include empty tags
    std::string tagData = translateSubtree(tag);
    // wrap the thing, ignoring whitespace
    size_t wsPos = tagData.find_last_not_of("\n\t ");
    if (wsPos != std::string::npos && wsPos != tagData.size() - 1)
      translatedComment += arg + tagData.substr(0, wsPos + 1) + arg + tagData.substr(wsPos + 1);
    else
      translatedComment += arg + tagData + arg;
  }
}

void PyDocConverter::handleDoxyHtmlTag(DoxygenEntity &tag, std::string &translatedComment, const std::string &arg) {
  std::string htmlTagArgs = tag.data;
  if (htmlTagArgs == "/") {
    // end html tag, for example "</ul>
    // translatedComment += "</" + arg.substr(1) + ">";
  } else {
    translatedComment += arg + htmlTagArgs;
  }
}

void PyDocConverter::handleDoxyHtmlTagNoParam(DoxygenEntity &tag, std::string &translatedComment, const std::string &arg) {
  std::string htmlTagArgs = tag.data;
  if (htmlTagArgs == "/") {
    // end html tag, for example "</ul>
  } else {
    translatedComment += arg;
  }
}

void PyDocConverter::handleDoxyHtmlTag_A(DoxygenEntity &tag, std::string &translatedComment, const std::string &arg) {
  std::string htmlTagArgs = tag.data;
  if (htmlTagArgs == "/") {
    // end html tag, "</a>
    translatedComment += " (" + m_url + ')';
    m_url.clear();
  } else {
    m_url.clear();
    size_t pos = htmlTagArgs.find('=');
    if (pos != string::npos) {
      m_url = htmlTagArgs.substr(pos + 1);
    }
    translatedComment += arg;
  }
}

void PyDocConverter::handleDoxyHtmlTag2(DoxygenEntity &tag, std::string &translatedComment, const std::string &arg) {
  std::string htmlTagArgs = tag.data;
  if (htmlTagArgs == "/") {
    // end html tag, for example "</em>
    translatedComment += arg;
  } else {
    translatedComment += arg;
  }
}

void PyDocConverter::handleDoxyHtmlTag_tr(DoxygenEntity &tag, std::string &translatedComment, const std::string &) {
  std::string htmlTagArgs = tag.data;
  size_t nlPos = translatedComment.rfind('\n');
  if (htmlTagArgs == "/") {
    // end tag, </tr> appends vertical table line '|'
    translatedComment += '|';
    if (nlPos != string::npos) {
      size_t startOfTableLinePos = translatedComment.find_first_not_of(" \t", nlPos + 1);
      if (startOfTableLinePos != string::npos) {
        m_tableLineLen = translatedComment.size() - startOfTableLinePos;
      }
    }
  } else {
    if (m_prevRowIsTH) {
      // if previous row contained <th> tag, add horizontal separator
      // but first get leading spaces, because they'll be needed for the next row
      size_t numLeadingSpaces = translatedComment.size() - nlPos - 1;

      translatedComment += string(m_tableLineLen, '-') + '\n';

      if (nlPos != string::npos) {
        translatedComment += string(numLeadingSpaces, ' ');
      }
      m_prevRowIsTH = false;
    }
  }
}

void PyDocConverter::handleDoxyHtmlTag_th(DoxygenEntity &tag, std::string &translatedComment, const std::string &) {
  std::string htmlTagArgs = tag.data;
  if (htmlTagArgs == "/") {
    // end tag, </th> is ignored
  } else {
    translatedComment += '|';
    m_prevRowIsTH = true;
  }
}

void PyDocConverter::handleDoxyHtmlTag_td(DoxygenEntity &tag, std::string &translatedComment, const std::string &) {
  std::string htmlTagArgs = tag.data;
  if (htmlTagArgs == "/") {
    // end tag, </td> is ignored
  } else {
    translatedComment += '|';
  }
}

void PyDocConverter::handleHtmlEntity(DoxygenEntity &, std::string &translatedComment, const std::string &arg) {
  // html entities
  translatedComment += arg;
}

void PyDocConverter::handleNewLine(DoxygenEntity &, std::string &translatedComment, const std::string &) {
  trimWhitespace(translatedComment);

  translatedComment += "\n";
  if (!m_indent.empty())
    translatedComment += m_indent;
}

String *PyDocConverter::makeDocumentation(Node *n) {
  String *documentation;
  std::string pyDocString;

  // store the node, we may need it later
  currentNode = n;

  // for overloaded functions we must concat documentation for underlying overloads
  if (Getattr(n, "sym:overloaded")) {
    // rewind to the first overload
    while (Getattr(n, "sym:previousSibling"))
      n = Getattr(n, "sym:previousSibling");

    std::vector<std::string> allDocumentation;

    // minimal indent of any documentation comments, not initialized yet
    size_t minIndent = static_cast<size_t>(-1);

    // for each real method (not a generated overload) append the documentation
    string oneDoc;
    while (n) {
      documentation = getDoxygenComment(n);
      if (!Swig_is_generated_overload(n) && documentation) {
        currentNode = n;
        if (GetFlag(n, "feature:doxygen:notranslate")) {
          String *comment = NewString("");
          Append(comment, documentation);
          Replaceall(comment, "\n *", "\n");
          oneDoc = Char(comment);
          Delete(comment);
        } else {
          std::list<DoxygenEntity> entityList = parser.createTree(n, documentation);
          DoxygenEntity root("root", entityList);

          oneDoc = translateSubtree(root);
        }

        // find the minimal indent of this documentation comment, we need to
        // ensure that the entire comment is indented by it to avoid the leading
        // parts of the other lines being simply discarded later
        const size_t oneIndent = determineIndent(oneDoc);
        if (oneIndent < minIndent)
          minIndent = oneIndent;

        allDocumentation.push_back(oneDoc);
      }
      n = Getattr(n, "sym:nextSibling");
    }

    // construct final documentation string
    if (allDocumentation.size() > 1) {
      string indentStr;
      if (minIndent != static_cast<size_t>(-1))
       indentStr.assign(minIndent, ' ');

      std::ostringstream concatDocString;
      for (size_t realOverloadCount = 0; realOverloadCount < allDocumentation.size(); realOverloadCount++) {
        if (realOverloadCount != 0) {
          // separate it from the preceding one.
          concatDocString << "\n" << indentStr << "|\n\n";
        }

        oneDoc = allDocumentation[realOverloadCount];
        trimWhitespace(oneDoc);
        concatDocString << indentStr << "*Overload " << (realOverloadCount + 1) << ":*\n" << oneDoc;
      }
      pyDocString = concatDocString.str();
    } else if (allDocumentation.size() == 1) {
      pyDocString = *(allDocumentation.begin());
    }
  }
  // for other nodes just process as normal
  else {
    documentation = getDoxygenComment(n);
    if (documentation != NULL) {
      if (GetFlag(n, "feature:doxygen:notranslate")) {
        String *comment = NewString("");
        Append(comment, documentation);
        Replaceall(comment, "\n *", "\n");
        pyDocString = Char(comment);
        Delete(comment);
      } else {
        std::list<DoxygenEntity> entityList = parser.createTree(n, documentation);
        DoxygenEntity root("root", entityList);
        pyDocString = translateSubtree(root);
      }
    }
  }

  // if we got something log the result
  if (!pyDocString.empty()) {

    // remove the last '\n' since additional one is added during writing to file
    eraseTrailingNewLine(pyDocString);

    // ensure that a blank line occurs before code or math blocks
    pyDocString = padCodeAndVerbatimBlocks(pyDocString);

    if (m_flags & debug_translator) {
      std::cout << "\n---RESULT IN PYDOC---" << std::endl;
      std::cout << pyDocString;
      std::cout << std::endl;
    }
  }

  return NewString(pyDocString.c_str());
}

