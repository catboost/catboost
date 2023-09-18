/* -----------------------------------------------------------------------------
 * This file is part of SWIG, which is licensed as a whole under version 3
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at https://www.swig.org/legal.html.
 *
 * javadoc.h
 *
 * Module to return documentation for nodes formatted for JavaDoc
 * ----------------------------------------------------------------------------- */

#ifndef SWIG_JAVADOC_H
#define SWIG_JAVADOC_H

#include "doxytranslator.h"
#include <map>

/*
 * A class to translate doxygen comments into JavaDoc style comments.
 */
class JavaDocConverter : public DoxygenTranslator {
public:
  JavaDocConverter(int flags = 0);
  String *makeDocumentation(Node *node);

protected:
  /*
   * Used to properly format JavaDoc-style command
   */
  std::string formatCommand(std::string unformattedLine, int indent);

  /*
   * Translate every entity in a tree.
   */
  std::string translateSubtree(DoxygenEntity &doxygenEntity);

  /*
   * Translate one entity with the appropriate handler, according
   * to the tagHandlers
   */
  void translateEntity(DoxygenEntity &tag, std::string &translatedComment);

  /*
   * Fix all endlines location, etc
   */
  int shiftEndlinesUpTree(DoxygenEntity &root, int level = 0);

  /*
   * Convert params in link-objects and references
   */
  std::string convertLink(std::string linkObject);

  /*
   * Typedef for the function that handles one tag
   * arg - some string argument to easily pass it through lookup table
   */
  typedef void (JavaDocConverter::*tagHandler) (DoxygenEntity &tag, std::string &translatedComment, std::string &arg);

  /**
   * Copies verbatim args of the tag to output, used for commands like \f$, ...
   */
  void handleTagVerbatim(DoxygenEntity &tag, std::string &translatedComment, std::string &arg);

  /** Creates anchor link. */
  void handleTagAnchor(DoxygenEntity &tag, std::string &translatedComment, std::string &arg);

  /*
   * Wrap the command data with the html tag
   * arg - html tag, with no braces
   */
  void handleTagHtml(DoxygenEntity &tag, std::string &translatedComment, std::string &arg);

  /* Handles HTML tags recognized by Doxygen, like <A ...>, <ul>, <table>, ... */
  void handleDoxyHtmlTag(DoxygenEntity &tag, std::string &translatedComment, std::string &arg);

  /* Handles HTML entities recognized by Doxygen, like &lt;, &copy;, ... */
  void handleHtmlEntity(DoxygenEntity &tag, std::string &translatedComment, std::string &arg);

  /*
   * Just prints new line
   */
  void handleNewLine(DoxygenEntity &tag, std::string &translatedComment, std::string &arg);

  /*
   * Print the name of tag to the output, used for escape-commands
   * arg - html-escaped variant, if not provided the command data is used
   */
  void handleTagChar(DoxygenEntity &tag, std::string &translatedComment, std::string &arg);

  /*
   * Do not translate and print as-is
   * arg - the new tag name, if it needs to be renamed
   */
  void handleTagSame(DoxygenEntity &tag, std::string &translatedComment, std::string &arg);

  /*
   * Print only the content and strip original tag
   */
  void handleParagraph(DoxygenEntity &tag, std::string &translatedComment, std::string &arg);

  /*
   * Print only data part of code
   */
  void handlePlainString(DoxygenEntity &tag, std::string &translatedComment, std::string &arg);

  /*
   * Print extended Javadoc command, like {@code ...} or {@literal ...}
   * arg - command name
   */
  void handleTagExtended(DoxygenEntity &tag, std::string &translatedComment, std::string &arg);

  /*
   * Print the if-elseif-else-endif section
   */
  void handleTagIf(DoxygenEntity &tag, std::string &translatedComment, std::string &arg);

  /*
   * Prints the specified message, than the contents of the tag
   * arg - message
   */
  void handleTagMessage(DoxygenEntity &tag, std::string &translatedComment, std::string &arg);

  /*
   * Insert <img src=... /> tag if the 'format' field is specified as 'html'
   */
  void handleTagImage(DoxygenEntity &tag, std::string &translatedComment, std::string &arg);

  /*
   * Insert <p alt='title'>...</p>
   */
  void handleTagPar(DoxygenEntity &tag, std::string &translatedComment, std::string &arg);

  /*
   * Insert \@param command, if it is really a function param
   */
  void handleTagParam(DoxygenEntity &tag, std::string &translatedComment, std::string &arg);

  /*
   * Writes link for \ref tag. 
   */
  void handleTagRef(DoxygenEntity &tag, std::string &translatedComment, std::string &);

  /*
   * Insert {@link...} command, and handle all the <link-object>s correctly
   * (like converting types of params, etc)
   */
  void handleTagLink(DoxygenEntity &tag, std::string &translatedComment, std::string &arg);

  /*
   * Insert @see command, and handle all the <link-object>s correctly
   * (like converting types of params, etc)
   */
  void handleTagSee(DoxygenEntity &tag, std::string &translatedComment, std::string &arg);

private:
  Node *currentNode;
  // this contains the handler pointer and one string argument
  static std::map<std::string, std::pair<tagHandler, std::string> > tagHandlers;
  void fillStaticTables();

  bool paramExists(std::string param);
  std::string indentAndInsertAsterisks(const std::string &doc);

  void addError(int warningType, const std::string &message);
};

#endif
