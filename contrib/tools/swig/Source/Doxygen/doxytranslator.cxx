/* -----------------------------------------------------------------------------
 * This file is part of SWIG, which is licensed as a whole under version 3
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * doxytranslator.cxx
 *
 * Module to return documentation for nodes formatted for various documentation
 * systems.
 * ----------------------------------------------------------------------------- */

#include "doxytranslator.h"

DoxygenTranslator::DoxygenTranslator(int flags) : m_flags(flags), parser((flags &debug_parser) != 0) {
}


DoxygenTranslator::~DoxygenTranslator() {
}


bool DoxygenTranslator::hasDocumentation(Node *node) {
  return getDoxygenComment(node) != NULL;
}


String *DoxygenTranslator::getDoxygenComment(Node *node) {
  return Getattr(node, "doxygen");
}

/**
 * Indent all lines in the comment by given indentation string
 */
void DoxygenTranslator::extraIndentation(String *comment, const_String_or_char_ptr indentationString) {
  if (indentationString || Len(indentationString) > 0) {
    int len = Len(comment);
    bool trailing_newline = len > 0 && *(Char(comment) + len - 1) == '\n';
    Insert(comment, 0, indentationString);
    String *replace = NewStringf("\n%s", indentationString);
    Replaceall(comment, "\n", replace);
    if (trailing_newline) {
      len = Len(comment);
      Delslice(comment, len - 2, len); // Remove added trailing spaces on last line
    }
    Delete(replace);
  }
}

String *DoxygenTranslator::getDocumentation(Node *node, const_String_or_char_ptr indentationString) {

  if (!hasDocumentation(node)) {
    return NewString("");
  }

  String *documentation = makeDocumentation(node);
  extraIndentation(documentation, indentationString);
  return documentation;
}


void DoxygenTranslator::printTree(const DoxygenEntityList &entityList) {

  for (DoxygenEntityListCIt p = entityList.begin(); p != entityList.end(); p++) {
    p->printEntity(0);
  }
}
