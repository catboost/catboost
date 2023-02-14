/* -----------------------------------------------------------------------------
 * This file is part of SWIG, which is licensed as a whole under version 3
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * doxytranslator.h
 *
 * Module to return documentation for nodes formatted for various documentation
 * systems.
 * ----------------------------------------------------------------------------- */

#ifndef DOXYGENTRANSLATOR_H_
#define DOXYGENTRANSLATOR_H_

#include "swig.h"
#include "doxyentity.h"
#include "doxyparser.h"
#include <list>
#include <string>


/*
 * This is a base class for translator classes. It defines the basic interface
 * for translators, which convert Doxygen comments into alternative formats for
 * target languages.
 */
class DoxygenTranslator {
public:
  /*
   * Bit flags for the translator ctor.
   *
   * Derived classes may define additional flags.
   */
  enum {
    // Use DoxygenParser in "noisy" mode.
    debug_parser = 1,

    // Output results of translating Doxygen comments.
    debug_translator = 2
  };

  /*
   * Constructor
   */
  DoxygenTranslator(int flags = 0);

  /*
   * Virtual destructor.
   */
  virtual ~DoxygenTranslator();

  /*
   * Return the documentation for a given node formated for the correct 
   * documentation system.
   */
  String *getDocumentation(Node *node, const_String_or_char_ptr indentationString);

  /*
   * Returns truem is the specified node has comment attached.
   */
  bool hasDocumentation(Node *node);

  /*
   * Get original comment string in Doxygen-format.
   */
  String *getDoxygenComment(Node *node);

protected:
  // The flags passed to the ctor.
  const int m_flags;

  DoxygenParser parser;

  /*
   * Returns the documentation formatted for a target language.
   */
  virtual String *makeDocumentation(Node *node) = 0;

  /*
   * Prints the details of a parsed entity list to stdout (for debugging).
   */
  void printTree(const DoxygenEntityList &entityList);

  void extraIndentation(String *comment, const_String_or_char_ptr indentationString);
};

#endif
