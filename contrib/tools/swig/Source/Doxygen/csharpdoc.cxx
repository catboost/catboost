/* -----------------------------------------------------------------------------
 * This file is part of SWIG, which is licensed as a whole under version 3
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * csharpdoc.cxx
 *
 * Module to return documentation for nodes formatted for CSharpDoc
 * ----------------------------------------------------------------------------- */

#include "csharpdoc.h"
#include "doxyparser.h"
#include <sstream>
#include <string>
#include <vector>
#include <iostream>

#include "swigmod.h"

// define static tables, they are filled in CSharpDocConverter's constructor
CSharpDocConverter::TagHandlersMap CSharpDocConverter::tagHandlers;

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
};

static void replaceAll(std::string &src, const std::string &token, const std::string &replace) {
  std::string::size_type pos = src.find(token);

  while (pos != std::string::npos) {
    src.replace(pos, token.size(), replace);
    pos = src.find(token, pos + replace.size());
  }
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

// Erase the first character in the string if it is a newline
static void eraseAllNewLine(string &str) {
  for (size_t i = 0; i < str.size(); i++) {
    // if the character is a newline character
    if (str[i] == '\n') {
      // remove the character
      str.erase(i, 1);
      // decrement the index to account for the removed character
      i--;
    }
  }
}

// Erase last characters in the string if it is a newline or a space
static void eraseTrailingSpaceNewLines(string &s) {
  while (!s.empty() && (s[s.size() - 1] == '\n' || s[s.size() - 1] == ' '))
    s.erase(s.size() - 1);
}

// escape some characters which cannot appear as it in C# comments
static void escapeSpecificCharacters(string &str) {
  for (size_t i = 0; i < str.size(); i++) {
    if (str[i] == '<') {
      str.replace(i, 1, "&lt;");
    } else if (str[i] == '>') {
      str.replace(i, 1, "&gt;");
    } else if (str[i] == '&') {
      str.replace(i, 1, "&amp;");
    }
  }
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

/* static */
CSharpDocConverter::TagHandlersMap::mapped_type CSharpDocConverter::make_handler(tagHandler handler) {
  return make_pair(handler, std::string());
}

/* static */
CSharpDocConverter::TagHandlersMap::mapped_type CSharpDocConverter::make_handler(tagHandler handler, const char *arg) {
  return make_pair(handler, arg);
}

void CSharpDocConverter::fillStaticTables() {
  if (tagHandlers.size()) // fill only once
    return;


  tagHandlers["a"] = make_handler(&CSharpDocConverter::handleTagWrap, "*");
  tagHandlers["b"] = make_handler(&CSharpDocConverter::handleTagWrap, "**");
  // \c command is translated as single quotes around next word
  tagHandlers["c"] = make_handler(&CSharpDocConverter::handleTagWrap, "``");
  tagHandlers["cite"] = make_handler(&CSharpDocConverter::handleTagWrap, "'");
  tagHandlers["e"] = make_handler(&CSharpDocConverter::handleTagWrap, "*");
  // these commands insert just a single char, some of them need to be escaped
  tagHandlers["$"] = make_handler(&CSharpDocConverter::handleTagChar);
  tagHandlers["@"] = make_handler(&CSharpDocConverter::handleTagChar);
  tagHandlers["\\"] = make_handler(&CSharpDocConverter::handleTagChar);
  tagHandlers["<"] = make_handler(&CSharpDocConverter::handleTagCharReplace, "&lt;");
  tagHandlers[">"] = make_handler(&CSharpDocConverter::handleTagCharReplace, "&gt;");
  tagHandlers["&"] = make_handler(&CSharpDocConverter::handleTagCharReplace, "&amp;");
  tagHandlers["#"] = make_handler(&CSharpDocConverter::handleTagChar);
  tagHandlers["%"] = make_handler(&CSharpDocConverter::handleTagChar);
  tagHandlers["~"] = make_handler(&CSharpDocConverter::handleTagChar);
  tagHandlers["\""] = make_handler(&CSharpDocConverter::handleTagChar);
  tagHandlers["."] = make_handler(&CSharpDocConverter::handleTagChar);
  tagHandlers["::"] = make_handler(&CSharpDocConverter::handleTagChar);
  // these commands are stripped out, and only their content is printed
  tagHandlers["attention"] = make_handler(&CSharpDocConverter::handleParagraph, "remarks");
  tagHandlers["author"] = make_handler(&CSharpDocConverter::handleTagWord, "Author");
  tagHandlers["authors"] = make_handler(&CSharpDocConverter::handleTagWord, "Author");
  tagHandlers["brief"] = make_handler(&CSharpDocConverter::handleSummary);
  tagHandlers["bug"] = make_handler(&CSharpDocConverter::handleTagWord, "Bug:");
  tagHandlers["code"] = make_handler(&CSharpDocConverter::handleCode);
  tagHandlers["copyright"] = make_handler(&CSharpDocConverter::handleParagraph, "remarks");
  tagHandlers["date"] = make_handler(&CSharpDocConverter::handleTagWord, "Date");
  tagHandlers["deprecated"] = make_handler(&CSharpDocConverter::handleTagWord, "Deprecated");
  tagHandlers["details"] = make_handler(&CSharpDocConverter::handleParagraph, "remarks");
  tagHandlers["em"] = make_handler(&CSharpDocConverter::handleTagWrap, "*");
  tagHandlers["example"] = make_handler(&CSharpDocConverter::handleTagWord, "Example");
  tagHandlers["exception"] = tagHandlers["throw"] = tagHandlers["throws"] = make_handler(&CSharpDocConverter::handleTagException);
  tagHandlers["htmlonly"] = make_handler(&CSharpDocConverter::handleNotHandled);
  tagHandlers["invariant"] = make_handler(&CSharpDocConverter::handleNotHandled);
  tagHandlers["latexonly"] = make_handler(&CSharpDocConverter::handleNotHandled);
  tagHandlers["link"] = make_handler(&CSharpDocConverter::handleNotHandled);
  tagHandlers["manonly"] = make_handler(&CSharpDocConverter::handleNotHandled);
  tagHandlers["note"] = make_handler(&CSharpDocConverter::handleNotHandled);
  tagHandlers["p"] = make_handler(&CSharpDocConverter::handleTagWrap, "``");
  tagHandlers["partofdescription"] = make_handler(&CSharpDocConverter::handleNotHandled);
  tagHandlers["rtfonly"] = make_handler(&CSharpDocConverter::handleNotHandled);
  tagHandlers["remark"] = make_handler(&CSharpDocConverter::handleParagraph, "remarks");
  tagHandlers["remarks"] = make_handler(&CSharpDocConverter::handleParagraph, "remarks");
  tagHandlers["sa"] = make_handler(&CSharpDocConverter::handleTagSee);
  tagHandlers["see"] = make_handler(&CSharpDocConverter::handleTagSee);
  tagHandlers["since"] = make_handler(&CSharpDocConverter::handleNotHandled);
  tagHandlers["short"] = make_handler(&CSharpDocConverter::handleNotHandled);
  tagHandlers["todo"] = make_handler(&CSharpDocConverter::handleTagWord, "TODO");
  tagHandlers["version"] = make_handler(&CSharpDocConverter::handleTagWord, "Version");
  tagHandlers["verbatim"] = make_handler(&CSharpDocConverter::handleVerbatimBlock);
  tagHandlers["warning"] = make_handler(&CSharpDocConverter::handleLine, "remarks");
  tagHandlers["xmlonly"] = make_handler(&CSharpDocConverter::handleNotHandled);
  // these commands have special handlers
  tagHandlers["arg"] = make_handler(&CSharpDocConverter::handleAddList);
  tagHandlers["cond"] = make_handler(&CSharpDocConverter::handleIgnore);
  tagHandlers["else"] = make_handler(&CSharpDocConverter::handleIgnore);
  tagHandlers["elseif"] = make_handler(&CSharpDocConverter::handleIgnore);
  tagHandlers["endcond"] = make_handler(&CSharpDocConverter::handleIgnore);
  tagHandlers["if"] = make_handler(&CSharpDocConverter::handleIgnore);
  tagHandlers["ifnot"] = make_handler(&CSharpDocConverter::handleIgnore);
  tagHandlers["image"] = make_handler(&CSharpDocConverter::handleIgnore);
  tagHandlers["li"] = make_handler(&CSharpDocConverter::handleIgnore);
  tagHandlers["overload"] = make_handler(&CSharpDocConverter::handleIgnore);

  tagHandlers["par"] = make_handler(&CSharpDocConverter::handleTagWord, "Title");
  tagHandlers["param"] = tagHandlers["tparam"] = make_handler(&CSharpDocConverter::handleTagParam);
  tagHandlers["ref"] = make_handler(&CSharpDocConverter::handleTagRef);
  tagHandlers["result"] = tagHandlers["return"] = tagHandlers["returns"] = make_handler(&CSharpDocConverter::handleTagReturn);

  // this command just prints its contents
  // (it is internal command of swig's parser, contains plain text)
  tagHandlers["plainstd::string"] = make_handler(&CSharpDocConverter::handlePlainString);
  tagHandlers["plainstd::endl"] = make_handler(&CSharpDocConverter::handleNewLine);
  tagHandlers["n"] = make_handler(&CSharpDocConverter::handleNewLine);

  // \f commands output literal Latex formula, which is still better than nothing.
  tagHandlers["f$"] = tagHandlers["f["] = tagHandlers["f{"] = make_handler(&CSharpDocConverter::handleMath);

  // HTML tags
  tagHandlers["<a"] = make_handler(&CSharpDocConverter::handleDoxyHtmlTag_A);
  tagHandlers["<b"] = make_handler(&CSharpDocConverter::handleDoxyHtmlTag2, "**");
  tagHandlers["<blockquote"] = make_handler(&CSharpDocConverter::handleDoxyHtmlTag_A, "Quote: ");
  tagHandlers["<body"] = make_handler(&CSharpDocConverter::handleDoxyHtmlTag);
  tagHandlers["<br"] = make_handler(&CSharpDocConverter::handleDoxyHtmlTag, "\n");

  // there is no formatting for this tag as it was deprecated in HTML 4.01 and
  // not used in HTML 5
  tagHandlers["<center"] = make_handler(&CSharpDocConverter::handleDoxyHtmlTag);
  tagHandlers["<caption"] = make_handler(&CSharpDocConverter::handleDoxyHtmlTag);
  tagHandlers["<code"] = make_handler(&CSharpDocConverter::handleDoxyHtmlTag2, "``");

  tagHandlers["<dl"] = make_handler(&CSharpDocConverter::handleDoxyHtmlTag);
  tagHandlers["<dd"] = make_handler(&CSharpDocConverter::handleDoxyHtmlTag, "    ");
  tagHandlers["<dt"] = make_handler(&CSharpDocConverter::handleDoxyHtmlTag);

  tagHandlers["<dfn"] = make_handler(&CSharpDocConverter::handleDoxyHtmlTag);
  tagHandlers["<div"] = make_handler(&CSharpDocConverter::handleDoxyHtmlTag);
  tagHandlers["<em"] = make_handler(&CSharpDocConverter::handleDoxyHtmlTag2, "**");
  tagHandlers["<form"] = make_handler(&CSharpDocConverter::handleDoxyHtmlTag);
  tagHandlers["<hr"] = make_handler(&CSharpDocConverter::handleDoxyHtmlTag, "--------------------------------------------------------------------\n");
  tagHandlers["<h1"] = make_handler(&CSharpDocConverter::handleDoxyHtmlTag, "# ");
  tagHandlers["<h2"] = make_handler(&CSharpDocConverter::handleDoxyHtmlTag, "## ");
  tagHandlers["<h3"] = make_handler(&CSharpDocConverter::handleDoxyHtmlTag, "### ");
  tagHandlers["<i"] = make_handler(&CSharpDocConverter::handleDoxyHtmlTag2, "*");
  tagHandlers["<input"] = make_handler(&CSharpDocConverter::handleDoxyHtmlTag);
  tagHandlers["<img"] = make_handler(&CSharpDocConverter::handleDoxyHtmlTag, "Image:");
  tagHandlers["<li"] = make_handler(&CSharpDocConverter::handleDoxyHtmlTag, "* ");
  tagHandlers["<meta"] = make_handler(&CSharpDocConverter::handleDoxyHtmlTag);
  tagHandlers["<multicol"] = make_handler(&CSharpDocConverter::handleDoxyHtmlTag);
  tagHandlers["<ol"] = make_handler(&CSharpDocConverter::handleDoxyHtmlTag);
  tagHandlers["<p"] = make_handler(&CSharpDocConverter::handleDoxyHtmlTag, "\n");
  tagHandlers["<pre"] = make_handler(&CSharpDocConverter::handleDoxyHtmlTag);
  tagHandlers["<small"] = make_handler(&CSharpDocConverter::handleDoxyHtmlTag);
  tagHandlers["<span"] = make_handler(&CSharpDocConverter::handleDoxyHtmlTag2, "'");
  tagHandlers["<strong"] = make_handler(&CSharpDocConverter::handleDoxyHtmlTag2, "**");

  // make a space between text and super/sub script.
  tagHandlers["<sub"] = make_handler(&CSharpDocConverter::handleDoxyHtmlTag, " ");
  tagHandlers["<sup"] = make_handler(&CSharpDocConverter::handleDoxyHtmlTag, " ");

  tagHandlers["<table"] = make_handler(&CSharpDocConverter::handleDoxyHtmlTagNoParam);
  tagHandlers["<td"] = make_handler(&CSharpDocConverter::handleDoxyHtmlTag_td);
  tagHandlers["<th"] = make_handler(&CSharpDocConverter::handleDoxyHtmlTag_th);
  tagHandlers["<tr"] = make_handler(&CSharpDocConverter::handleDoxyHtmlTag_tr);
  tagHandlers["<tt"] = make_handler(&CSharpDocConverter::handleDoxyHtmlTag);
  tagHandlers["<kbd"] = make_handler(&CSharpDocConverter::handleDoxyHtmlTag);
  tagHandlers["<ul"] = make_handler(&CSharpDocConverter::handleDoxyHtmlTag);
  tagHandlers["<var"] = make_handler(&CSharpDocConverter::handleDoxyHtmlTag2, "*");

  // HTML entities
  tagHandlers["&copy"] = make_handler(&CSharpDocConverter::handleHtmlEntity, "(C)");
  tagHandlers["&trade"] = make_handler(&CSharpDocConverter::handleHtmlEntity, " TM");
  tagHandlers["&reg"] = make_handler(&CSharpDocConverter::handleHtmlEntity, "(R)");
  tagHandlers["&lt"] = make_handler(&CSharpDocConverter::handleHtmlEntity, "<");
  tagHandlers["&gt"] = make_handler(&CSharpDocConverter::handleHtmlEntity, ">");
  tagHandlers["&amp"] = make_handler(&CSharpDocConverter::handleHtmlEntity, "&");
  tagHandlers["&apos"] = make_handler(&CSharpDocConverter::handleHtmlEntity, "'");
  tagHandlers["&quot"] = make_handler(&CSharpDocConverter::handleHtmlEntity, "\"");
  tagHandlers["&lsquo"] = make_handler(&CSharpDocConverter::handleHtmlEntity, "`");
  tagHandlers["&rsquo"] = make_handler(&CSharpDocConverter::handleHtmlEntity, "'");
  tagHandlers["&ldquo"] = make_handler(&CSharpDocConverter::handleHtmlEntity, "\"");
  tagHandlers["&rdquo"] = make_handler(&CSharpDocConverter::handleHtmlEntity, "\"");
  tagHandlers["&ndash"] = make_handler(&CSharpDocConverter::handleHtmlEntity, "-");
  tagHandlers["&mdash"] = make_handler(&CSharpDocConverter::handleHtmlEntity, "--");
  tagHandlers["&nbsp"] = make_handler(&CSharpDocConverter::handleHtmlEntity, " ");
  tagHandlers["&times"] = make_handler(&CSharpDocConverter::handleHtmlEntity, "x");
  tagHandlers["&minus"] = make_handler(&CSharpDocConverter::handleHtmlEntity, "-");
  tagHandlers["&sdot"] = make_handler(&CSharpDocConverter::handleHtmlEntity, ".");
  tagHandlers["&sim"] = make_handler(&CSharpDocConverter::handleHtmlEntity, "~");
  tagHandlers["&le"] = make_handler(&CSharpDocConverter::handleHtmlEntity, "<=");
  tagHandlers["&ge"] = make_handler(&CSharpDocConverter::handleHtmlEntity, ">=");
  tagHandlers["&larr"] = make_handler(&CSharpDocConverter::handleHtmlEntity, "<--");
  tagHandlers["&rarr"] = make_handler(&CSharpDocConverter::handleHtmlEntity, "-->");
}

CSharpDocConverter::CSharpDocConverter(int flags):
DoxygenTranslator(flags), m_tableLineLen(0), m_prevRowIsTH(false) {
  fillStaticTables();
}

// Return the type as it should appear in the output documentation.
static std::string getCSharpDocType(Node *n, const_String_or_char_ptr lname = "") {
  std::string type;

  String *s = Swig_typemap_lookup("doctype", n, lname, 0);
  if (!s) {
    if (String *t = Getattr(n, "type"))
      s = SwigType_str(t, "");
  }
  /////////////////

  if (!s)
    return type;

  type = Char(s);

  Delete(s);

  return type;
}

std::string CSharpDocConverter::getParamType(std::string param) {
  std::string type;

  ParmList *plist = CopyParmList(Getattr(currentNode, "parms"));
  for (Parm *p = plist; p; p = nextSibling(p)) {
    String *pname = Getattr(p, "name");
    if (pname && Char(pname) == param) {
      type = getCSharpDocType(p, pname);
      break;
    }
  }
  Delete(plist);
  return type;
}

std::string CSharpDocConverter::getParamValue(std::string param) {
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

/**
 * Returns true, if the given parameter exists in the current node
 * (for example param is a name of function parameter). If feature
 * 'doxygen:nostripparams' is set, then this method always returns
 * true - parameters are copied to output regardless of presence in
 * function params list.
 */
bool CSharpDocConverter::paramExists(std::string param) {

  if (GetFlag(currentNode, "feature:doxygen:nostripparams")) {
    return true;
  }

  ParmList *plist = CopyParmList(Getattr(currentNode, "parms"));

  for (Parm *p = plist; p;) {

    if (Getattr(p, "name") && Char(Getattr(p, "name")) == param) {
      return true;
    }
    /* doesn't seem to work always: in some cases (especially for 'self' parameters)
     * tmap:in is present, but tmap:in:next is not and so this code skips all the parameters
     */
    //p = Getattr(p, "tmap:in") ? Getattr(p, "tmap:in:next") : nextSibling(p);
    p = nextSibling(p);
  }

  Delete(plist);

  return false;
}

std::string CSharpDocConverter::translateSubtree(DoxygenEntity &doxygenEntity) {
  std::string translatedComment;

  if (doxygenEntity.isLeaf)
    return translatedComment;

  std::string currentSection;
  std::list<DoxygenEntity>::iterator p = doxygenEntity.entityList.begin();
  while (p != doxygenEntity.entityList.end()) {
    translateEntity(*p, translatedComment);
    translateSubtree(*p);
    p++;
  }

  return translatedComment;
}

void CSharpDocConverter::translateEntity(DoxygenEntity &doxyEntity, std::string &translatedComment) {
  // check if we have needed handler and call it
  std::map<std::string, std::pair<tagHandler, std::string> >::iterator it;
  it = tagHandlers.find(getBaseCommand(doxyEntity.typeOfEntity));
  if (it != tagHandlers.end())
    (this->*(it->second.first)) (doxyEntity, translatedComment, it->second.second);
}

void CSharpDocConverter::handleIgnore(DoxygenEntity &tag, std::string &translatedComment, const std::string &) {
  if (tag.entityList.size()) {
    tag.entityList.pop_front();
  }

  translatedComment += translateSubtree(tag);
}

void CSharpDocConverter::handleSummary(DoxygenEntity &tag, std::string &translatedComment, const std::string &) {

  translatedComment += "<summary>";
  std::string summary = translateSubtree(tag);

  eraseAllNewLine(summary);
  trimWhitespace(summary);
  // remove final newlines
  eraseTrailingSpaceNewLines(summary);
  escapeSpecificCharacters(summary);

  translatedComment += summary;


  translatedComment += "</summary>";
  translatedComment += "\n";
}

void CSharpDocConverter::handleLine(DoxygenEntity &tag, std::string &translatedComment, const std::string &tagName) {

  translatedComment += "<" + tagName + ">";
  if (tag.entityList.size()) {
    translatedComment += tag.entityList.begin()->data;
    tag.entityList.pop_front();
  }
  translatedComment += "</" + tagName + ">";
}


void CSharpDocConverter::handleNotHandled(DoxygenEntity &tag, std::string &translatedComment, const std::string &) {

  std::string paragraph = translateSubtree(tag);

  eraseLeadingNewLine(paragraph);
  eraseTrailingSpaceNewLines(paragraph);
  trimWhitespace(paragraph);
  escapeSpecificCharacters(paragraph);
  translatedComment += paragraph;
  translatedComment += "\n";
}

void CSharpDocConverter::handleAddList(DoxygenEntity &tag, std::string &translatedComment, const std::string &) {
  std::string listItem = translateSubtree(tag);
  eraseAllNewLine(listItem);

  translatedComment += "* ";
  translatedComment += listItem;
  translatedComment += "\n";
}


void CSharpDocConverter::handleParagraph(DoxygenEntity &tag, std::string &translatedComment, const std::string &tagName) {
  translatedComment += "<";
  translatedComment += tagName;
  translatedComment += ">";

  std::string paragraph = translateSubtree(tag);

  eraseAllNewLine(paragraph);
  trimWhitespace(paragraph);
  eraseTrailingSpaceNewLines(paragraph);
  escapeSpecificCharacters(paragraph);

  translatedComment += paragraph;

  translatedComment += "</";
  translatedComment += tagName;
  translatedComment += ">\n";
}

void CSharpDocConverter::handleVerbatimBlock(DoxygenEntity &tag, std::string &translatedComment, const std::string &) {
  string verb = translateSubtree(tag);

  eraseLeadingNewLine(verb);

  // Remove the last newline to prevent doubling the newline already present after \endverbatim
  trimWhitespace(verb); // Needed to catch trailing newline below
  eraseTrailingSpaceNewLines(verb);
  escapeSpecificCharacters(verb);

  translatedComment += verb;
}

void CSharpDocConverter::handleMath(DoxygenEntity &tag, std::string &translatedComment, const std::string &arg) {
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

void CSharpDocConverter::handleCode(DoxygenEntity &tag, std::string &translatedComment, const std::string &arg) {
  IndentGuard indent(translatedComment, m_indent);

  trimWhitespace(translatedComment);

  translatedComment += "<code>";

  std::string code;
  handleTagVerbatim(tag, code, arg);

  // Try and remove leading newline, which is present for block \code
  // command:
  escapeSpecificCharacters(code);
  eraseLeadingNewLine(code);
  trimWhitespace(code);

  // Check for python doctest blocks, and treat them specially:
  bool isDocTestBlock = false;
  size_t startPos;
  // ">>>" would normally appear at the beginning, but doxygen comment
  // style may have space in front, so skip leading whitespace
  if ((startPos = code.find_first_not_of(" \t")) != string::npos && code.substr(startPos, 3) == ">>>")
    isDocTestBlock = true;

  string codeIndent;
  if (!isDocTestBlock) {
    // Use the current indent for the code-block line itself.
    translatedComment += indent.getFirstLineIndent();

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
  eraseTrailingSpaceNewLines(translatedComment);

  translatedComment += "</code>";
  translatedComment += "\n";
}

void CSharpDocConverter::handlePlainString(DoxygenEntity &tag, std::string &translatedComment, const std::string &) {
  translatedComment += tag.data;
}

void CSharpDocConverter::handleTagVerbatim(DoxygenEntity &tag, std::string &translatedComment, const std::string &arg) {
  translatedComment += arg;
  for (DoxygenEntityListCIt it = tag.entityList.begin(); it != tag.entityList.end(); it++) {
    translatedComment += it->data;
  }
}

void CSharpDocConverter::handleTagMessage(DoxygenEntity &tag, std::string &translatedComment, const std::string &arg) {
  translatedComment += arg;
  handleParagraph(tag, translatedComment);
  translatedComment += "\">\n";
}

void CSharpDocConverter::handleTagSee(DoxygenEntity &tag, std::string &translatedComment, const std::string &) {
  translatedComment += "<seealso cref=\"";
  std::string seeAlso = translateSubtree(tag);
  escapeSpecificCharacters(seeAlso);

  // Remove parameter list
  // Alternative would be to try and convert them into C# types similar to Java implementation
  std::string::size_type lbrace = seeAlso.find('(');
  if (lbrace != std::string::npos)
    seeAlso.erase(lbrace);

  replaceAll(seeAlso, "::", ".");
  eraseTrailingSpaceNewLines(seeAlso);

  translatedComment += seeAlso;
  translatedComment += "\"/>\n";
}

void CSharpDocConverter::handleTagCharReplace(DoxygenEntity &, std::string &translatedComment, const std::string &arg) {
  translatedComment += arg;
}

void CSharpDocConverter::handleTagChar(DoxygenEntity &tag, std::string &translatedComment, const std::string &) {
  translatedComment += tag.typeOfEntity;
}

void CSharpDocConverter::handleTagIf(DoxygenEntity &tag, std::string &translatedComment, const std::string &arg) {
  translatedComment += arg;
  if (tag.entityList.size()) {
    translatedComment += tag.entityList.begin()->data;
    tag.entityList.pop_front();
    translatedComment += " {" + translateSubtree(tag) + "}";
  }
}

void CSharpDocConverter::handleTagWord(DoxygenEntity &tag, std::string &translatedComment, const std::string &arg) {
  translatedComment += arg + ": ";
  if (tag.entityList.size())
    translatedComment += tag.entityList.begin()->data;
  tag.entityList.pop_front();
  translatedComment += translateSubtree(tag);
  translatedComment += "\n";
}

void CSharpDocConverter::handleTagImage(DoxygenEntity &tag, std::string &translatedComment, const std::string &) {
  if (tag.entityList.size() < 2)
    return;
  tag.entityList.pop_front();
  translatedComment += "Image: ";
  translatedComment += tag.entityList.begin()->data;
  tag.entityList.pop_front();
  if (tag.entityList.size())
    translatedComment += "(" + tag.entityList.begin()->data + ")";
}

void CSharpDocConverter::handleTagParam(DoxygenEntity &tag, std::string &translatedComment, const std::string &) {

  if (tag.entityList.size() < 2)
    return;

  if (!paramExists(tag.entityList.begin()->data))
    return;

  IndentGuard indent(translatedComment, m_indent);

  DoxygenEntity paramNameEntity = *tag.entityList.begin();
  tag.entityList.pop_front();

  const std::string &paramName = paramNameEntity.data;

  const std::string paramValue = getParamValue(paramName);

  translatedComment += "<param name=\"" + paramName + "\">";

  translatedComment += translateSubtree(tag);
  eraseTrailingSpaceNewLines(translatedComment);

  translatedComment += "</param> \n";
}


void CSharpDocConverter::handleTagReturn(DoxygenEntity &tag, std::string &translatedComment, const std::string &) {
  IndentGuard indent(translatedComment, m_indent);

  translatedComment += "<returns>";
  translatedComment += translateSubtree(tag);
  eraseTrailingSpaceNewLines(translatedComment);
  translatedComment += "</returns> \n";
}


void CSharpDocConverter::handleTagException(DoxygenEntity &tag, std::string &translatedComment, const std::string &) {
  IndentGuard indent(translatedComment, m_indent);

  DoxygenEntity paramNameEntity = *tag.entityList.begin();
  tag.entityList.pop_front();

  const std::string &paramName = paramNameEntity.data;

  const std::string paramType = getParamType(paramName);
  const std::string paramValue = getParamValue(paramName);

  translatedComment += "<exception cref=\"" + paramName + "\">";

  translatedComment += translateSubtree(tag);
  eraseTrailingSpaceNewLines(translatedComment);

  translatedComment += "</exception> \n";
}


void CSharpDocConverter::handleTagRef(DoxygenEntity &tag, std::string &translatedComment, const std::string &) {

  if (!tag.entityList.size())
    return;

  string anchor = tag.entityList.begin()->data;
  tag.entityList.pop_front();
  string anchorText = anchor;

  size_t pos = anchorText.find('#');
  if (pos != string::npos) {
    anchorText = anchorText.substr(pos + 1);
  }

  if (!tag.entityList.empty()) {
    anchorText = tag.entityList.begin()->data;
  }
  translatedComment += "\\ref " + anchorText;
}


void CSharpDocConverter::handleTagWrap(DoxygenEntity &tag, std::string &translatedComment, const std::string &arg) {
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

void CSharpDocConverter::handleDoxyHtmlTag(DoxygenEntity &tag, std::string &translatedComment, const std::string &arg) {
  std::string htmlTagArgs = tag.data;
  if (htmlTagArgs == "/") {
    // end html tag, for example "</ul>
    // translatedComment += "</" + arg.substr(1) + ">";
  } else {
    translatedComment += arg + htmlTagArgs;
  }
}

void CSharpDocConverter::handleDoxyHtmlTagNoParam(DoxygenEntity &tag, std::string &translatedComment, const std::string &arg) {
  std::string htmlTagArgs = tag.data;
  if (htmlTagArgs == "/") {
    // end html tag, for example "</ul>
  } else {
    translatedComment += arg;
  }
}

void CSharpDocConverter::handleDoxyHtmlTag_A(DoxygenEntity &tag, std::string &translatedComment, const std::string &arg) {
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

void CSharpDocConverter::handleDoxyHtmlTag2(DoxygenEntity &tag, std::string &translatedComment, const std::string &arg) {
  std::string htmlTagArgs = tag.data;
  if (htmlTagArgs == "/") {
    // end html tag, for example "</em>
    translatedComment += arg;
  } else {
    translatedComment += arg;
  }
}

void CSharpDocConverter::handleDoxyHtmlTag_tr(DoxygenEntity &tag, std::string &translatedComment, const std::string &) {
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
	translatedComment += string (numLeadingSpaces, ' ');
      }
      m_prevRowIsTH = false;
    }
  }
}

void CSharpDocConverter::handleDoxyHtmlTag_th(DoxygenEntity &tag, std::string &translatedComment, const std::string &) {
  std::string htmlTagArgs = tag.data;
  if (htmlTagArgs == "/") {
    // end tag, </th> is ignored
  } else {
    translatedComment += '|';
    m_prevRowIsTH = true;
  }
}

void CSharpDocConverter::handleDoxyHtmlTag_td(DoxygenEntity &tag, std::string &translatedComment, const std::string &) {
  std::string htmlTagArgs = tag.data;
  if (htmlTagArgs == "/") {
    // end tag, </td> is ignored
  } else {
    translatedComment += '|';
  }
}

void CSharpDocConverter::handleHtmlEntity(DoxygenEntity &, std::string &translatedComment, const std::string &arg) {
  // html entities
  translatedComment += arg;
}

void CSharpDocConverter::handleNewLine(DoxygenEntity &, std::string &translatedComment, const std::string &) {
  trimWhitespace(translatedComment);

  translatedComment += "\n";

  if (!m_indent.empty())
    translatedComment += m_indent;
}

String *CSharpDocConverter::makeDocumentation(Node *n) {
  String *documentation;
  std::string csharpDocString;

  // store the node, we may need it later
  currentNode = n;

  documentation = getDoxygenComment(n);
  if (documentation != NULL) {
    if (GetFlag(n, "feature:doxygen:notranslate")) {
      String *comment = NewString("");
      Append(comment, documentation);
      Replaceall(comment, "\n *", "\n");
      csharpDocString = Char(comment);
      Delete(comment);
    } else {
      std::list<DoxygenEntity> entityList = parser.createTree(n, documentation);
      DoxygenEntity root("root", entityList);
      csharpDocString = translateSubtree(root);
    }
  }

  // if we got something log the result
  if (!csharpDocString.empty()) {

    // remove the last spaces and '\n' since additional one is added during writing to file
    eraseTrailingSpaceNewLines(csharpDocString);

    // ensure that a blank line occurs before code or math blocks
    csharpDocString = padCodeAndVerbatimBlocks(csharpDocString);

    if (m_flags & debug_translator) {
      std::cout << "\n---RESULT IN CSHARPDOC---" << std::endl;
      std::cout << csharpDocString;
      std::cout << std::endl;
    }
  }

  return NewString(csharpDocString.c_str());
}
