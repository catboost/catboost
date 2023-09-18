/* -----------------------------------------------------------------------------
 * This file is part of SWIG, which is licensed as a whole under version 3
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at https://www.swig.org/legal.html.
 *
 * doxyparser.h
 * ----------------------------------------------------------------------------- */

#ifndef SWIG_DOXYPARSER_H
#define SWIG_DOXYPARSER_H
#include <string>
#include <list>
#include <map>
#include <vector>
#include <set>

#include "swig.h"

#include "doxyentity.h"

// Utility function to return the base part of a command that may
// include options, e.g. param[in] -> param
std::string getBaseCommand(const std::string &cmd);


class DoxygenParser {
private:

  enum DoxyCommandEnum {
    NONE = -1,
    SIMPLECOMMAND,
    COMMANDWORD,
    COMMANDLINE,
    COMMANDPARAGRAPH,
    COMMANDENDCOMMAND,
    COMMANDWORDPARAGRAPH,
    COMMANDWORDLINE,
    COMMANDWORDOWORDWORD,
    COMMANDOWORD,
    COMMANDERRORTHROW,
    COMMANDUNIQUE,
    COMMAND_HTML,
    COMMAND_HTML_ENTITY,
    COMMAND_ALIAS,
    COMMAND_IGNORE,
    END_LINE,
    PARAGRAPH_END,
    PLAINSTRING,
    COMMAND
  };


  /** This class contains parts of Doxygen comment as a token. */
  class Token {
  public:
    DoxyCommandEnum m_tokenType;
    std::string m_tokenString; /* the data , such as param for @param */

    Token(DoxyCommandEnum tType, std::string tString) : m_tokenType(tType), m_tokenString(tString) {
    }
    
    std::string toString() const {
      switch (m_tokenType) {
      case END_LINE:
        return "{END OF LINE}";
      case PARAGRAPH_END:
        return "{END OF PARAGRAPH}";
      case PLAINSTRING:
        return "{PLAINSTRING :" + m_tokenString + "}";
      case COMMAND:
        return "{COMMAND : " + m_tokenString + "}";
      default:
        return "";
      }
    }
  };


  typedef std::vector<Token> TokenList;
  typedef TokenList::const_iterator TokenListCIt;
  typedef TokenList::iterator TokenListIt;

  TokenList m_tokenList;
  TokenListCIt m_tokenListIt;

  typedef std::map<std::string, DoxyCommandEnum> DoxyCommandsMap;
  typedef DoxyCommandsMap::iterator DoxyCommandsMapIt;

  /*
   * Map of Doxygen commands to determine if a string is a
   * command and how it needs to be parsed
   */
  static DoxyCommandsMap doxygenCommands;
  static std::set<std::string> doxygenSectionIndicators;

  bool m_isVerbatimText; // used to handle \htmlonly and \verbatim commands
  bool m_isInQuotedString;

  Node *m_node;
  std::string m_fileName;
  int m_fileLineNo;

  /*
   * Return the end command for a command appearing in "ignore" feature or empty
   * string if this is a simple command and not a block one.
   */
  std::string getIgnoreFeatureEndCommand(const std::string &theCommand) const;

  /*
   * Helper for getting the value of doxygen:ignore feature or its argument.
   */
  String *getIgnoreFeature(const std::string &theCommand, const char *argument = NULL) const;

  /*
   * Whether to print lots of debug info during parsing
   */
  bool noisy;

  /*
   *Changes a std::string to all lower case
   */
  std::string stringToLower(const std::string &stringToConvert);

  /* 
   * isSectionIndicator returns a boolean if the command is a section indicator
   * This is a helper method for finding the end of a paragraph
   * by Doxygen's terms
   */
  bool isSectionIndicator(const std::string &smallString);
  /*
   * Determines how a command should be handled (what group it belongs to
   * for parsing rules
   */
  DoxyCommandEnum commandBelongs(const std::string &theCommand);

  /*
   *prints the parse tree
   */
  void printTree(const std::list<DoxygenEntity> &rootList);

  /**
   * Returns true if the next token is end of line token. This is important
   * when single word commands like \c are at the end of line.
   */
  bool isEndOfLine();

  /**
   * Skips spaces, tabs, and end of line tokens.
   */
  void skipWhitespaceTokens();

  /**
   * Removes all spaces and tabs from beginning end end of string.
   */
  std::string trim(const std::string &text);

  /*
   * Returns string of the next token if the next token is PLAINSTRING. Returns
   * empty string otherwise.
   */
  std::string getNextToken();

  /*
   * Returns the next word ON THE CURRENT LINE ONLY
   * if a new line is encountered, returns a blank std::string.
   * Updates the iterator if successful.
   */
  std::string getNextWord();

  /*
   * Returns the next word, which is not necessarily on the same line.
   * Updates the iterator if successful.
   */
  std::string getNextWordInComment();

  /* 
   * Returns the location of the end of the line as
   * an iterator.
   */
  TokenListCIt getOneLine(const TokenList &tokList);

  /*
   * Returns a properly formatted std::string
   * up til ANY command or end of line is encountered.
   */
  std::string getStringTilCommand(const TokenList &tokList);

  /*
   * Returns a properly formatted std::string
   * up til the command specified is encountered
   */
  //TODO check that this behaves properly for formulas
  std::string getStringTilEndCommand(const std::string &theCommand, const TokenList &tokList);

  /*
   * Returns the end of a Paragraph as an iterator-
   * Paragraph is defined in Doxygen to be a paragraph of text
   * separated by either a structural command or a blank line
   */
  TokenListCIt getEndOfParagraph(const TokenList &tokList);

  /*
   * Returns the end of a section, defined as the first blank line OR first
   * encounter of the same command. Example of this behaviour is \arg.
   * If no end is encountered, returns the last token of the std::list.
   */
  TokenListCIt getEndOfSection(const std::string &theCommand, const TokenList &tokList);

  /*
   * This method is for returning the end of a specific form of doxygen command
   * that begins with a \command and ends in \endcommand
   * such as \code and \endcode. The proper usage is
   * progressTilEndCommand("endcode", tokenList);
   * If the end is never encountered, it returns the end of the std::list.
   */
  TokenListCIt getEndCommand(const std::string &theCommand, const TokenList &tokList);
  /*
   * A special method for commands such as \arg that end at the end of a
   * paragraph OR when another \arg is encountered
  //TODO getTilAnyCommand
  TokenListCIt getTilAnyCommand(const std::string &theCommand, const TokenList &tokList);
   */

  /**
   * This methods skips end of line token, if it is the next token to be
   * processed. It is called with comment commands which have args till the
   * end of line, such as 'addtogroup' or 'addindex'.
   * It is up to translator to specific language to decide whether
   * to insert eol or not. For example, if a command is ignored in target
   * language, new lines may make formatting ugly (Python).
   */
  void skipEndOfLine();

  /*
   * Method for Adding a Simple Command
   * Format: @command
   * Plain commands, such as newline etc, they contain no other data
   *  \n \\ \@ \& \$ \# \< \> \% \{ \}
   */
  void addSimpleCommand(const std::string &theCommand, DoxygenEntityList &doxyList);
  /*
   * CommandWord
   * Format: @command <word>
   * Commands with a single WORD after then such as @b
   * "a", "b", "c", "e", "em", "p", "def", "enum", "example", "package", 
   * "relates", "namespace", "relatesalso","anchor", "dontinclude", "include",
   * "includelineno"
   */
  void addCommandWord(const std::string &theCommand, const TokenList &tokList, DoxygenEntityList &doxyList);
  /*
   * CommandLine
   * Format: @command (line)
   * Commands with a single LINE after then such as @var
   * "addindex", "fn", "name", "line", "var", "skipline", "typedef", "skip",
   * "until", "property"
   */
  void addCommandLine(const std::string &theCommand, const TokenList &tokList, DoxygenEntityList &doxyList);
  /*
   * CommandParagraph
   * Format: @command {paragraph}
   * Commands with a single paragraph after then such as @return
   * "return", "remarks", "since", "test", "sa", "see", "pre", "post",
   * "details", "invariant", "deprecated", "date", "note", "warning",
   * "version", "todo", "bug", "attention", "brief", "arg", "author"
   */
  void addCommandParagraph(const std::string &theCommand, const TokenList &tokList, DoxygenEntityList &doxyList);
  /*
   * Command EndCommand
   * Format: @command and ends at @endcommand
   * Commands that take in a block of text such as @code:
   * "code", "dot", "msc", "f$", "f[", "f{environment}{", "htmlonly",
   * "latexonly", "manonly", "verbatim", "xmlonly", "cond", "if", "ifnot",
   * "link"
   * Returns 1 if success, 0 if the endcommand is never encountered.
   */
  void addCommandEndCommand(const std::string &theCommand, const TokenList &tokList, DoxygenEntityList &doxyList);
  /*
   * CommandWordParagraph
   * Format: @command <word> {paragraph}
   * Commands such as param
   * "param", "tparam", "throw", "throws", "retval", "exception"
   */
  void addCommandWordParagraph(const std::string &theCommand, const TokenList &tokList, DoxygenEntityList &doxyList);
  /*
   * CommandWordLine
   * Format: @command <word> (line)
   * Commands such as param
   * "page", "subsection", "subsubsection", "section", "paragraph", "defgroup"
   */
  void addCommandWordLine(const std::string &theCommand, const TokenList &tokList, DoxygenEntityList &doxyList);
  /*
   * Command Word Optional Word Optional Word
   * Format: @command <word> [<header-file>] [<header-name>]
   * Commands such as class
   * "category", "class", "protocol", "interface", "struct", "union"
   */
  void addCommandWordOWordOWord(const std::string &theCommand, const TokenList &tokList, DoxygenEntityList &doxyList);
  /*
   * Command Optional Word
   * Format: @command [<word>]
   * Commands such as dir
   * "dir", "file", "cond"
   */
  void addCommandOWord(const std::string &theCommand, const TokenList &tokList, DoxygenEntityList &doxyList);

  /*
   * Commands that should not be encountered (such as PHP only)
   * goes til the end of line then returns
   */
  void addCommandErrorThrow(const std::string &theCommand, const TokenList &tokList, DoxygenEntityList &doxyList);

  void addCommandHtml(const std::string &theCommand, const TokenList &tokList, DoxygenEntityList &doxyList);

  void addCommandHtmlEntity(const std::string &theCommand, const TokenList &tokList, DoxygenEntityList &doxyList);

  /*
   *Adds the unique commands- different process for each unique command
   */
  void addCommandUnique(const std::string &theCommand, const TokenList &tokList, DoxygenEntityList &doxyList);

  /*
   * Replace the given command with its predefined alias expansion.
   */
  void aliasCommand(const std::string &theCommand, const TokenList &tokList, DoxygenEntityList &doxyList);

  /*
   * Simply ignore the given command, possibly with the word following it or
   * until the matching end command.
   */
  void ignoreCommand(const std::string &theCommand, const TokenList &tokList, DoxygenEntityList &doxyList);

  /* 
   * The actual "meat" of the doxygen parser. Calls the correct addCommand...()
   * function.
   */
  void addCommand(const std::string &commandString, const TokenList &tokList, DoxygenEntityList &doxyList);

  DoxygenEntityList parse(TokenListCIt endParsingIndex, const TokenList &tokList, bool root = false);

  /*
   * Fill static doxygenCommands and sectionIndicators containers
   */
  void fillTables();

  /** Processes comment when \htmlonly and \verbatim commands are encountered. */
  size_t processVerbatimText(size_t pos, const std::string &line);

  bool processEscapedChars(size_t &pos, const std::string &line);
  void processWordCommands(size_t &pos, const std::string &line);
  void processHtmlTags(size_t &pos, const std::string &line);
  void processHtmlEntities(size_t &pos, const std::string &line);


  /** Processes comment outside \htmlonly and \verbatim commands. */
  size_t processNormalComment(size_t pos, const std::string &line);

  void tokenizeDoxygenComment(const std::string &doxygenComment, const std::string &fileName, int fileLine);
  void printList();
  void printListError(int warningType, const std::string &message);

  typedef std::vector<std::string> StringVector;
  typedef StringVector::const_iterator StringVectorCIt;

  StringVector split(const std::string &text, char separator);
  bool isStartOfDoxyCommentChar(char c);
  bool addDoxyCommand(DoxygenParser::TokenList &tokList, const std::string &cmd);

public:
  DoxygenParser(bool noisy = false);
  virtual ~DoxygenParser();
  DoxygenEntityList createTree(Node *node, String *documentation);
};

#endif
