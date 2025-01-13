/* -----------------------------------------------------------------------------
 * This file is part of SWIG, which is licensed as a whole under version 3
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at https://www.swig.org/legal.html.
 *
 * doxyparser.cxx
 * ----------------------------------------------------------------------------- */

#include "doxyparser.h"
#include "doxycommands.h"
#include "swig.h"
#include "swigwarn.h"

#include <iostream>
#include <vector>

using std::string;
using std::cout;
using std::endl;

// This constant defines the (only) characters valid inside a Doxygen "word".
// It includes some unusual ones because of the commands such as \f[, \f{, \f],
// \f} and \f$.
static const char *DOXYGEN_WORD_CHARS = "abcdefghijklmnopqrstuvwxyz" "ABCDEFGHIJKLMNOPQRSTUVWXYZ" "0123456789" "$[]{}";

// Define static class members
DoxygenParser::DoxyCommandsMap DoxygenParser::doxygenCommands;
std::set<std::string> DoxygenParser::doxygenSectionIndicators;

const int TOKENSPERLINE = 8; //change this to change the printing behaviour of the token list
const std::string END_HTML_TAG_MARK("/");

std::string getBaseCommand(const std::string &cmd) {
  if (cmd.substr(0,5) == "param")
    return "param";
  else if (cmd.substr(0,4) == "code")
    return "code";
  else
    return cmd;
}

// Find the first position beyond the word command.  Extra logic is
// used to avoid putting the characters "," and "." in
// DOXYGEN_WORD_CHARS.
static size_t getEndOfWordCommand(const std::string &line, size_t pos) {
  size_t endOfWordPos = line.find_first_not_of(DOXYGEN_WORD_CHARS, pos);
  if (line.substr(pos, 6) == "param[")
    // include ",", which can appear in param[in,out]
    endOfWordPos = line.find_first_not_of(string(DOXYGEN_WORD_CHARS)+ ",", pos);  
  else if (line.substr(pos, 5) == "code{")
    // include ".", which can appear in e.g. code{.py}
    endOfWordPos = line.find_first_not_of(string(DOXYGEN_WORD_CHARS)+ ".", pos);
  return endOfWordPos;
}


DoxygenParser::DoxygenParser(bool noisy) : noisy(noisy) {
  fillTables();
}

DoxygenParser::~DoxygenParser() {
}

void DoxygenParser::fillTables() {
  // run it only once
  if (doxygenCommands.size())
    return;

  // fill in tables with data from doxycommands.h
  for (int i = 0; i < simpleCommandsSize; i++)
    doxygenCommands[simpleCommands[i]] = SIMPLECOMMAND;

  for (int i = 0; i < commandWordsSize; i++)
    doxygenCommands[commandWords[i]] = COMMANDWORD;

  for (int i = 0; i < commandLinesSize; i++)
    doxygenCommands[commandLines[i]] = COMMANDLINE;

  for (int i = 0; i < commandParagraphSize; i++)
    doxygenCommands[commandParagraph[i]] = COMMANDPARAGRAPH;

  for (int i = 0; i < commandEndCommandsSize; i++)
    doxygenCommands[commandEndCommands[i]] = COMMANDENDCOMMAND;

  for (int i = 0; i < commandWordParagraphsSize; i++)
    doxygenCommands[commandWordParagraphs[i]] = COMMANDWORDPARAGRAPH;

  for (int i = 0; i < commandWordLinesSize; i++)
    doxygenCommands[commandWordLines[i]] = COMMANDWORDLINE;

  for (int i = 0; i < commandWordOWordOWordsSize; i++)
    doxygenCommands[commandWordOWordOWords[i]] = COMMANDWORDOWORDWORD;

  for (int i = 0; i < commandOWordsSize; i++)
    doxygenCommands[commandOWords[i]] = COMMANDOWORD;

  for (int i = 0; i < commandErrorThrowingsSize; i++)
    doxygenCommands[commandErrorThrowings[i]] = COMMANDERRORTHROW;

  for (int i = 0; i < commandUniquesSize; i++)
    doxygenCommands[commandUniques[i]] = COMMANDUNIQUE;

  for (int i = 0; i < commandHtmlSize; i++)
    doxygenCommands[commandHtml[i]] = COMMAND_HTML;

  for (int i = 0; i < commandHtmlEntitiesSize; i++)
    doxygenCommands[commandHtmlEntities[i]] = COMMAND_HTML_ENTITY;

  // fill section indicators command set
  for (int i = 0; i < sectionIndicatorsSize; i++)
    doxygenSectionIndicators.insert(sectionIndicators[i]);
}

std::string DoxygenParser::stringToLower(const std::string &stringToConvert) {

  string result(stringToConvert.size(), ' ');

  for (size_t i = 0; i < result.size(); i++) {
    result[i] = tolower(stringToConvert[i]);
  }

  return result;
}

bool DoxygenParser::isSectionIndicator(const std::string &smallString) {

  std::set<std::string>::iterator it = doxygenSectionIndicators.find(stringToLower(smallString));

  return it != doxygenSectionIndicators.end();
}

void DoxygenParser::printTree(const DoxygenEntityList &rootList) {
  DoxygenEntityList::const_iterator p = rootList.begin();
  while (p != rootList.end()) {
    (*p).printEntity(0);
    p++;
  }
}

DoxygenParser::DoxyCommandEnum DoxygenParser::commandBelongs(const std::string &theCommand) {
  DoxyCommandsMapIt it = doxygenCommands.find(stringToLower(getBaseCommand(theCommand)));

  if (it != doxygenCommands.end()) {
    return it->second;
  }
  // Check if this command is defined as an alias.
  if (Getattr(m_node, ("feature:doxygen:alias:" + theCommand).c_str())) {
    return COMMAND_ALIAS;
  }
  // Check if this command should be ignored.
  if (String *const ignore = getIgnoreFeature(theCommand)) {
    // Check that no value is specified for this feature ("1" is the implicit
    // one given to it by SWIG itself), we may use the value in the future, but
    // for now we only use the attributes.
    if (Strcmp(ignore, "1") != 0) {
      Swig_warning(WARN_PP_UNEXPECTED_TOKENS, m_fileName.c_str(), m_fileLineNo,
                   "Feature \"doxygen:ignore\" value ignored for Doxygen command \"%s\".\n", theCommand.c_str());
    }
    // Also ensure that the matching end command, if any, will be recognized.
    const string endCommand = getIgnoreFeatureEndCommand(theCommand);
    if (!endCommand.empty()) {
      Setattr(m_node, ("feature:doxygen:ignore:" + endCommand).c_str(), NewString("1"));
    }

    return COMMAND_IGNORE;
  }

  return NONE;
}

std::string DoxygenParser::trim(const std::string &text) {
  size_t start = text.find_first_not_of(" \t");
  size_t end = text.find_last_not_of(" \t");

  if (start == string::npos || start > end) {
    return "";
  }
  return text.substr(start, end - start + 1);
}

bool DoxygenParser::isEndOfLine() {
  if (m_tokenListIt == m_tokenList.end()) {
    return false;
  }
  Token nextToken = *m_tokenListIt;
  return nextToken.m_tokenType == END_LINE;
}

void DoxygenParser::skipWhitespaceTokens() {
  if (m_tokenListIt == m_tokenList.end()) {
    return;
  }

  while (m_tokenListIt != m_tokenList.end()
         && (m_tokenListIt->m_tokenType == END_LINE || trim(m_tokenListIt->m_tokenString).empty())) {

    m_tokenListIt++;
  }
}

std::string DoxygenParser::getNextToken() {

  if (m_tokenListIt == m_tokenList.end()) {
    return "";
  }

  if (m_tokenListIt->m_tokenType == PLAINSTRING) {
    return (m_tokenListIt++)->m_tokenString;
  }

  return "";
}

std::string DoxygenParser::getNextWord() {

  /*    if (m_tokenListIt == m_tokenList.end()) {
     return "";
     }
   */
  while (m_tokenListIt != m_tokenList.end()
         && (m_tokenListIt->m_tokenType == PLAINSTRING)) {
    // handle quoted strings as words
    string token = m_tokenListIt->m_tokenString;
    if (token == "\"") {

      string word = m_tokenListIt->m_tokenString;
      m_tokenListIt++;
      while (true) {
        string nextWord = getNextToken();
        if (nextWord.empty()) { // maybe report unterminated string error
          return word;
        }
        word += nextWord;
        if (nextWord == "\"") {
          return word;
        }
      }
    }

    string tokenStr = trim(m_tokenListIt->m_tokenString);
    m_tokenListIt++;
    if (!tokenStr.empty()) {
      return tokenStr;
    }
  }

  return "";
}

DoxygenParser::TokenListCIt DoxygenParser::getOneLine(const TokenList &tokList) {

  TokenListCIt endOfLineIt = m_tokenListIt;

  while (endOfLineIt != tokList.end()) {
    if (endOfLineIt->m_tokenType == END_LINE) {
      return endOfLineIt;
    }
    endOfLineIt++;
  }

  return tokList.end();
}

std::string DoxygenParser::getStringTilCommand(const TokenList &tokList) {

  if (m_tokenListIt == tokList.end()) {
    return "";
  }

  string description;

  while (m_tokenListIt->m_tokenType == PLAINSTRING) {
    const Token &currentToken = *m_tokenListIt++;
    if (currentToken.m_tokenType == PLAINSTRING) {
      description = description + currentToken.m_tokenString; // + " ";
    }
  }
  return description;
}

std::string DoxygenParser::getStringTilEndCommand(const std::string &theCommand, const TokenList &tokList) {

  if (m_tokenListIt == tokList.end()) {
    return "";
  }

  string description;
  while (m_tokenListIt != tokList.end()) {

    if (m_tokenListIt->m_tokenType == PLAINSTRING) {
      description += m_tokenListIt->m_tokenString;
    } else if (m_tokenListIt->m_tokenType == END_LINE) {
      description += "\n";
    } else if (m_tokenListIt->m_tokenString == theCommand) {
      m_tokenListIt++;
      return description;
    }

    m_tokenListIt++;
  }

  printListError(WARN_DOXYGEN_COMMAND_EXPECTED, "Expected Doxygen command: " + theCommand + ".");

  return description;
}

DoxygenParser::TokenListCIt DoxygenParser::getEndOfParagraph(const TokenList &tokList) {

  TokenListCIt endOfParagraph = m_tokenListIt;

  while (endOfParagraph != tokList.end()) {
    // If \code or \verbatim is encountered within a paragraph, then
    // go all the way to the end of that command, since the content
    // could contain empty lines that would appear to be paragraph
    // ends:
    if (endOfParagraph->m_tokenType == COMMAND &&
	(endOfParagraph->m_tokenString == "code" ||
	 endOfParagraph->m_tokenString == "verbatim")) {
      const string theCommand = endOfParagraph->m_tokenString;
      endOfParagraph = getEndCommand("end" + theCommand, tokList);
      endOfParagraph++; // Move after the end command
      return endOfParagraph;
    }
    if (endOfParagraph->m_tokenType == END_LINE) {
      endOfParagraph++;
      if (endOfParagraph != tokList.end()
          && endOfParagraph->m_tokenType == END_LINE) {
        endOfParagraph++;
        //cout << "ENCOUNTERED END OF PARA" << endl;
        return endOfParagraph;
      }

    } else if (endOfParagraph->m_tokenType == COMMAND) {

      if (isSectionIndicator(getBaseCommand(endOfParagraph->m_tokenString))) {
        return endOfParagraph;
      } else {
        endOfParagraph++;
      }

    } else if (endOfParagraph->m_tokenType == PLAINSTRING) {
      endOfParagraph++;
    } else {
      return tokList.end();
    }
  }

  return tokList.end();
}

DoxygenParser::TokenListCIt DoxygenParser::getEndOfSection(const std::string &theCommand, const TokenList &tokList) {

  TokenListCIt endOfParagraph = m_tokenListIt;

  while (endOfParagraph != tokList.end()) {
    if (endOfParagraph->m_tokenType == COMMAND) {
      if (theCommand == endOfParagraph->m_tokenString)
        return endOfParagraph;
      else
        endOfParagraph++;
    } else if (endOfParagraph->m_tokenType == PLAINSTRING) {
      endOfParagraph++;
    } else if (endOfParagraph->m_tokenType == END_LINE) {
      endOfParagraph++;
      if (endOfParagraph->m_tokenType == END_LINE) {
        endOfParagraph++;
        return endOfParagraph;
      }
    }
  }
  return tokList.end();
}

DoxygenParser::TokenListCIt DoxygenParser::getEndCommand(const std::string &theCommand, const TokenList &tokList) {

  TokenListCIt endOfCommand = m_tokenListIt;

  while (endOfCommand != tokList.end()) {
    endOfCommand++;
    if ((*endOfCommand).m_tokenType == COMMAND) {
      if (theCommand == (*endOfCommand).m_tokenString) {
        return endOfCommand;
      }
    }
  }
  //End command not found
  return tokList.end();
}

void DoxygenParser::skipEndOfLine() {
  if (m_tokenListIt != m_tokenList.end()
      && m_tokenListIt->m_tokenType == END_LINE) {
    m_tokenListIt++;
  }
}

void DoxygenParser::addSimpleCommand(const std::string &theCommand, DoxygenEntityList &doxyList) {
  if (noisy)
    cout << "Parsing " << theCommand << endl;

  doxyList.push_back(DoxygenEntity(theCommand));
}

void DoxygenParser::addCommandWord(const std::string &theCommand, const TokenList &, DoxygenEntityList &doxyList) {
  if (noisy)
    cout << "Parsing " << theCommand << endl;

  if (isEndOfLine()) {
    // handles cases when command is at the end of line (for example "\c\nreally"
    skipWhitespaceTokens();
    doxyList.push_back(DoxygenEntity("plainstd::endl"));
  }
  std::string name = getNextWord();
  if (!name.empty()) {
    DoxygenEntityList aNewList;
    aNewList.push_back(DoxygenEntity("plainstd::string", name));
    doxyList.push_back(DoxygenEntity(theCommand, aNewList));
  } else {
    printListError(WARN_DOXYGEN_COMMAND_ERROR, "Error parsing Doxygen command " + theCommand + ": No word followed the command. Command ignored.");
  }
}

void DoxygenParser::addCommandLine(const std::string &theCommand, const TokenList &tokList, DoxygenEntityList &doxyList) {
  if (noisy)
    cout << "Parsing " << theCommand << endl;
  TokenListCIt endOfLine = getOneLine(tokList);
  DoxygenEntityList aNewList = parse(endOfLine, tokList);
  doxyList.push_back(DoxygenEntity(theCommand, aNewList));
  skipEndOfLine();
}

void DoxygenParser::addCommandParagraph(const std::string &theCommand, const TokenList &tokList, DoxygenEntityList &doxyList) {
  if (noisy)
    cout << "Parsing " << theCommand << endl;

  TokenListCIt endOfParagraph = getEndOfParagraph(tokList);
  DoxygenEntityList aNewList;
  aNewList = parse(endOfParagraph, tokList);
  doxyList.push_back(DoxygenEntity(theCommand, aNewList));
}

void DoxygenParser::addCommandEndCommand(const std::string &theCommand, const TokenList &tokList, DoxygenEntityList &doxyList) {
  if (noisy)
    cout << "Parsing " << theCommand << endl;
  TokenListCIt endCommand = getEndCommand("end" + theCommand, tokList);
  if (endCommand == tokList.end()) {
    printListError(WARN_DOXYGEN_COMMAND_EXPECTED, "Expected Doxygen command: end" + theCommand + ".");
    return;
  }
  DoxygenEntityList aNewList;
  aNewList = parse(endCommand, tokList);
  m_tokenListIt++;
  doxyList.push_back(DoxygenEntity(theCommand, aNewList));
}

void DoxygenParser::addCommandWordParagraph(const std::string &theCommand, const TokenList &tokList, DoxygenEntityList &doxyList) {
  if (noisy)
    cout << "Parsing " << theCommand << endl;

  std::string name = getNextWord();

  if (name.empty()) {
    printListError(WARN_DOXYGEN_COMMAND_ERROR, "Error parsing Doxygen command " + theCommand + ": No word followed the command. Command ignored.");
    return;
  }
  TokenListCIt endOfParagraph = getEndOfParagraph(tokList);
  DoxygenEntityList aNewList;
  aNewList = parse(endOfParagraph, tokList);
  aNewList.push_front(DoxygenEntity("plainstd::string", name));
  doxyList.push_back(DoxygenEntity(theCommand, aNewList));
}

void DoxygenParser::addCommandWordLine(const std::string &theCommand, const TokenList &tokList, DoxygenEntityList &doxyList) {
  if (noisy)
    cout << "Parsing " << theCommand << endl;
  std::string name = getNextWord();
  if (name.empty()) {
    printListError(WARN_DOXYGEN_COMMAND_ERROR, "Error parsing Doxygen command " + theCommand + ": No word followed the command. Command ignored.");
    return;
  }

  TokenListCIt endOfLine = getOneLine(tokList);
  DoxygenEntityList aNewList;
  aNewList = parse(endOfLine, tokList);
  aNewList.push_front(DoxygenEntity("plainstd::string", name));
  doxyList.push_back(DoxygenEntity(theCommand, aNewList));
  //else cout << "No line followed " << theCommand <<  " command. Not added" << endl;
}

void DoxygenParser::addCommandWordOWordOWord(const std::string &theCommand, const TokenList &, DoxygenEntityList &doxyList) {
  if (noisy)
    cout << "Parsing " << theCommand << endl;

  std::string name = getNextWord();
  if (name.empty()) {
    printListError(WARN_DOXYGEN_COMMAND_ERROR, "Error parsing Doxygen command " + theCommand + ": No word followed the command. Command ignored.");
    return;
  }
  std::string headerfile = getNextWord();
  std::string headername = getNextWord();
  DoxygenEntityList aNewList;
  aNewList.push_back(DoxygenEntity("plainstd::string", name));
  if (!headerfile.empty())
    aNewList.push_back(DoxygenEntity("plainstd::string", headerfile));
  if (!headername.empty())
    aNewList.push_back(DoxygenEntity("plainstd::string", headername));
  doxyList.push_back(DoxygenEntity(theCommand, aNewList));
}

void DoxygenParser::addCommandOWord(const std::string &theCommand, const TokenList &, DoxygenEntityList &doxyList) {
  if (noisy)
    cout << "Parsing " << theCommand << endl;

  std::string name = getNextWord();
  DoxygenEntityList aNewList;
  aNewList.push_back(DoxygenEntity("plainstd::string", name));
  doxyList.push_back(DoxygenEntity(theCommand, aNewList));
}

void DoxygenParser::addCommandErrorThrow(const std::string &theCommand, const TokenList &tokList, DoxygenEntityList &) {

  printListError(WARN_DOXYGEN_COMMAND_ERROR, "Error parsing Doxygen command " + theCommand + ": Unexpectedly encountered this command.");
  m_tokenListIt = getOneLine(tokList);
}

void DoxygenParser::addCommandHtml(const std::string &theCommand, const TokenList &, DoxygenEntityList &doxyList) {
  if (noisy)
    cout << "Parsing " << theCommand << endl;

  std::string htmlTagArgs = getNextToken();
  doxyList.push_back(DoxygenEntity(theCommand, htmlTagArgs));
}

void DoxygenParser::addCommandHtmlEntity(const std::string &theCommand, const TokenList &, DoxygenEntityList &doxyList) {
  if (noisy)
    cout << "Parsing " << theCommand << endl;

  DoxygenEntityList aNewList;
  doxyList.push_back(DoxygenEntity(theCommand, aNewList));
}

void DoxygenParser::addCommandUnique(const std::string &theCommand, const TokenList &tokList, DoxygenEntityList &doxyList) {

  static std::map<std::string, std::string> endCommands;
  DoxygenEntityList aNewList;
  if (theCommand == "arg" || theCommand == "li") {
    TokenListCIt endOfSection = getEndOfSection(theCommand, tokList);
    DoxygenEntityList aNewList;
    aNewList = parse(endOfSection, tokList);
    doxyList.push_back(DoxygenEntity(theCommand, aNewList));
  }
  // \xrefitem <key> "(heading)" "(std::list title)" {text}
  else if (theCommand == "xrefitem") {
    if (noisy)
      cout << "Parsing " << theCommand << endl;
    std::string key = getNextWord();
    if (key.empty()) {
      printListError(WARN_DOXYGEN_COMMAND_ERROR, "Error parsing Doxygen command " + theCommand + ": No key followed the command. Command ignored.");
      return;
    }
    std::string heading = getNextWord();
    if (key.empty()) {
      printListError(WARN_DOXYGEN_COMMAND_ERROR, "Error parsing Doxygen command " + theCommand + ": No heading followed the command. Command ignored.");
      return;
    }
    std::string title = getNextWord();
    if (title.empty()) {
      printListError(WARN_DOXYGEN_COMMAND_ERROR, "Error parsing Doxygen command " + theCommand + ": No title followed the command. Command ignored.");
      return;
    }
    TokenListCIt endOfParagraph = getEndOfParagraph(tokList);
    aNewList = parse(endOfParagraph, tokList);
    aNewList.push_front(DoxygenEntity("plainstd::string", title));
    aNewList.push_front(DoxygenEntity("plainstd::string", heading));
    aNewList.push_front(DoxygenEntity("plainstd::string", key));
    doxyList.push_back(DoxygenEntity(theCommand, aNewList));
  }
  // \ingroup (<groupname> [<groupname> <groupname>])
  else if (theCommand == "ingroup") {
    std::string name = getNextWord();
    aNewList.push_back(DoxygenEntity("plainstd::string", name));
    name = getNextWord();
    if (!name.empty())
      aNewList.push_back(DoxygenEntity("plainstd::string", name));
    name = getNextWord();
    if (!name.empty())
      aNewList.push_back(DoxygenEntity("plainstd::string", name));
    doxyList.push_back(DoxygenEntity(theCommand, aNewList));
  }
  // \par [(paragraph title)] { paragraph }
  else if (theCommand == "par") {
    TokenListCIt endOfLine = getOneLine(tokList);
    aNewList = parse(endOfLine, tokList);
    DoxygenEntityList aNewList2;
    TokenListCIt endOfParagraph = getEndOfParagraph(tokList);
    aNewList2 = parse(endOfParagraph, tokList);
    aNewList.splice(aNewList.end(), aNewList2);
    doxyList.push_back(DoxygenEntity(theCommand, aNewList));
  }
  // \headerfile <header-file> [<header-name>]
  else if (theCommand == "headerfile") {
    DoxygenEntityList aNewList;
    std::string name = getNextWord();
    aNewList.push_back(DoxygenEntity("plainstd::string", name));
    name = getNextWord();
    if (!name.empty())
      aNewList.push_back(DoxygenEntity("plainstd::string", name));
    doxyList.push_back(DoxygenEntity(theCommand, aNewList));
  }
  // \overload [(function declaration)]
  else if (theCommand == "overload") {
    TokenListCIt endOfLine = getOneLine(tokList);
    if (endOfLine != m_tokenListIt) {
      DoxygenEntityList aNewList;
      aNewList = parse(endOfLine, tokList);
      doxyList.push_back(DoxygenEntity(theCommand, aNewList));
    } else
      doxyList.push_back(DoxygenEntity(theCommand));
  }
  // \weakgroup <name> [(title)]
  else if (theCommand == "weakgroup") {
    if (noisy)
      cout << "Parsing " << theCommand << endl;
    std::string name = getNextWord();
    if (name.empty()) {
      printListError(WARN_DOXYGEN_COMMAND_ERROR, "Error parsing Doxygen command " + theCommand + ": No word followed the command. Command ignored.");
      return;
    }
    DoxygenEntityList aNewList;
    TokenListCIt endOfLine = getOneLine(tokList);
    if (endOfLine != m_tokenListIt) {
      aNewList = parse(endOfLine, tokList);
    }
    aNewList.push_front(DoxygenEntity("plainstd::string", name));
    doxyList.push_back(DoxygenEntity(theCommand, aNewList));
  }
  // \ref <name> ["(text)"]
  else if (theCommand == "ref") {
    if (noisy)
      cout << "Parsing " << theCommand << endl;
    std::string name = getNextWord();
    if (name.empty()) {
      printListError(WARN_DOXYGEN_COMMAND_ERROR, "Error parsing Doxygen command " + theCommand + ": No key followed the command. Command ignored.");
      return;
    }
    DoxygenEntityList aNewList;
    aNewList.push_front(DoxygenEntity("plainstd::string", name));
    // TokenListCIt endOfLine = getOneLine(tokList);
    // if (endOfLine != m_tokenListIt) {
    //   aNewList = parse(endOfLine, tokList);
    //}
    TokenListCIt tmpIt = m_tokenListIt;
    std::string refTitle = getNextWord();
    // If title is following the ref tag, it must be quoted. Otherwise
    // doxy puts link on ref id.
    if (refTitle.size() > 1 && refTitle[0] == '"') {
      // remove quotes
      refTitle = refTitle.substr(1, refTitle.size() - 2);
      aNewList.push_back(DoxygenEntity("plainstd::string", refTitle));
    } else {
      // no quoted string is following, so we have to restore iterator
      m_tokenListIt = tmpIt;
    }
    doxyList.push_back(DoxygenEntity(theCommand, aNewList));
  }
  // \subpage <name> ["(text)"]
  else if (theCommand == "subpage") {
    if (noisy)
      cout << "Parsing " << theCommand << endl;
    std::string name = getNextWord();
    if (name.empty()) {
      printListError(WARN_DOXYGEN_COMMAND_ERROR, "Error parsing Doxygen command " + theCommand + ": No name followed the command. Command ignored.");
      return;
    }
    std::string text = getNextWord();
    aNewList.push_back(DoxygenEntity("plainstd::string", name));
    if (!text.empty())
      aNewList.push_back(DoxygenEntity("plainstd::string", text));
    doxyList.push_back(DoxygenEntity(theCommand, aNewList));
  }
  // \code ... \endcode
  // \verbatim ... \endverbatim
  // \dot dotcode \enddot
  // \msc msccode \endmsc
  // \f[ ... \f]
  // \f{ ... \f}
  // \f{env}{ ... \f}
  // \f$ ... \f$
  else if (getBaseCommand(theCommand) == "code" || theCommand == "verbatim"
           || theCommand == "dot" || theCommand == "msc" || theCommand == "f[" || theCommand == "f{" || theCommand == "f$") {
    if (!endCommands.size()) {
      // fill in static table of end commands
      endCommands["f["] = "f]";
      endCommands["f{"] = "f}";
      endCommands["f$"] = "f$";
    }
    if (noisy)
      cout << "Parsing " << theCommand << endl;

    std::string endCommand;
    std::map<std::string, std::string>::iterator it;
    it = endCommands.find(theCommand);
    if (it != endCommands.end())
      endCommand = it->second;
    else
      endCommand = "end" + getBaseCommand(theCommand);

    std::string content = getStringTilEndCommand(endCommand, tokList);
    aNewList.push_back(DoxygenEntity("plainstd::string", content));
    doxyList.push_back(DoxygenEntity(theCommand, aNewList));
  }
  // \dotfile <file> ["caption"]
  // \mscfile <file> ["caption"]
  else if (theCommand == "dotfile" || theCommand == "mscfile") {
    if (noisy)
      cout << "Parsing " << theCommand << endl;
    std::string file = getNextWord();
    if (file.empty()) {
      printListError(WARN_DOXYGEN_COMMAND_ERROR, "Error parsing Doxygen command " + theCommand + ": No file followed the command. Command ignored.");
      return;
    }
    std::string caption = getNextWord();
    aNewList.push_back(DoxygenEntity("plainstd::string", file));
    if (!caption.empty())
      aNewList.push_back(DoxygenEntity("plainstd::string", caption));
    doxyList.push_back(DoxygenEntity(theCommand, aNewList));
  }
  // \image <format> <file> ["caption"] [<sizeindication>=<size>]
  else if (theCommand == "image") {
    if (noisy)
      cout << "Parsing " << theCommand << endl;
    std::string format = getNextWord();
    if (format.empty()) {
      printListError(WARN_DOXYGEN_COMMAND_ERROR, "Error parsing Doxygen command " + theCommand + ": No format followed the command. Command ignored.");
      return;
    }
    std::string file = getNextWord();
    if (file.empty()) {
      printListError(WARN_DOXYGEN_COMMAND_ERROR, "Error parsing Doxygen command " + theCommand + ": No name followed the command. Command ignored.");
      return;
    }
    std::string caption = getNextWord();
    std::string size = getNextWord();

    DoxygenEntityList aNewList;
    aNewList.push_back(DoxygenEntity("plainstd::string", format));
    aNewList.push_back(DoxygenEntity("plainstd::string", file));
    if (!caption.empty())
      aNewList.push_back(DoxygenEntity("plainstd::string", caption));
    if (!size.empty())
      aNewList.push_back(DoxygenEntity("plainstd::string", size));
    doxyList.push_back(DoxygenEntity(theCommand, aNewList));
  }
  // \addtogroup <name> [(title)]
  else if (theCommand == "addtogroup") {
    if (noisy)
      cout << "Parsing " << theCommand << endl;
    std::string name = getNextWord();
    if (name.empty()) {
      printListError(WARN_DOXYGEN_COMMAND_ERROR, "Error parsing Doxygen command " + theCommand + ": There should be at least one word following the command. Command ignored.");
      return;
    }
    DoxygenEntityList aNewList;
    TokenListCIt endOfLine = getOneLine(tokList);
    if (endOfLine != m_tokenListIt) {
      aNewList = parse(endOfLine, tokList);
    }
    aNewList.push_front(DoxygenEntity("plainstd::string", name));
    doxyList.push_back(DoxygenEntity(theCommand, aNewList));
    skipEndOfLine();
  }
  // \if <cond> [\else ...] [\elseif <cond> ...] \endif
  else if (theCommand == "if" || theCommand == "ifnot" || theCommand == "else" || theCommand == "elseif") {
    if (noisy)
      cout << "Parsing " << theCommand << endl;

    std::string cond;
    bool skipEndif = false; // if true then we skip endif after parsing block of code
    bool needsCond = (theCommand == "if" || theCommand == "ifnot" || theCommand == "elseif");
    if (needsCond) {
      cond = getNextWord();
      if (cond.empty()) {
        printListError(WARN_DOXYGEN_COMMAND_ERROR, "Error parsing Doxygen command " + theCommand + ": No word followed the command. Command ignored.");
        return;
      }
    }

    int nestedCounter = 1;
    TokenListCIt endCommand = tokList.end();

    // go through the commands and find closing endif or else or elseif
    for (TokenListCIt it = m_tokenListIt; it != tokList.end(); it++) {
      if (it->m_tokenType == COMMAND) {
        if (it->m_tokenString == "if" || it->m_tokenString == "ifnot")
          nestedCounter++;
        else if (it->m_tokenString == "endif")
          nestedCounter--;
        if (nestedCounter == 1 && (it->m_tokenString == "else" || it->m_tokenString == "elseif")) { // else found
          endCommand = it;
          break;
        }
        if (nestedCounter == 0) { // endif found
          endCommand = it;
          skipEndif = true;
          break;
        }
      }
    }

    if (endCommand == tokList.end()) {
      printListError(WARN_DOXYGEN_COMMAND_EXPECTED, "Expected Doxygen command: endif.");
      return;
    }

    DoxygenEntityList aNewList;
    aNewList = parse(endCommand, tokList);
    if (skipEndif)
      m_tokenListIt++;
    if (needsCond)
      aNewList.push_front(DoxygenEntity("plainstd::string", cond));
    doxyList.push_back(DoxygenEntity(theCommand, aNewList));
  }
}

void DoxygenParser::aliasCommand(const std::string &theCommand, const TokenList &/* tokList */ , DoxygenEntityList &doxyList) {
  String *const alias = Getattr(m_node, ("feature:doxygen:alias:" + theCommand).c_str());
  if (!alias)
    return;

  doxyList.push_back(DoxygenEntity("plainstd::string", Char(alias)));
}

String *DoxygenParser::getIgnoreFeature(const std::string &theCommand, const char *argument) const {
  string feature_name = "feature:doxygen:ignore:" + theCommand;
  if (argument) {
    feature_name += ':';
    feature_name += argument;
  }

  return Getattr(m_node, feature_name.c_str());
}

string DoxygenParser::getIgnoreFeatureEndCommand(const std::string &theCommand) const {
  // We may be dealing either with a simple command or with the starting command
  // of a block, as indicated by the value of "range" starting with "end".
  string endCommand;
  if (String *const range = getIgnoreFeature(theCommand, "range")) {
    const char *const p = Char(range);
    if (strncmp(p, "end", 3) == 0) {
      if (p[3] == ':') {
        // Normally the end command name follows after the colon.
        endCommand = p + 4;
      } else if (p[3] == '\0') {
        // But it may be omitted in which case the default Doxygen convention of
        // using "something"/"endsomething" is used.
        endCommand = "end" + theCommand;
      }
    }
  }

  return endCommand;
}

void DoxygenParser::ignoreCommand(const std::string &theCommand, const TokenList &tokList, DoxygenEntityList &doxyList) {
  const string endCommand = getIgnoreFeatureEndCommand(theCommand);
  if (!endCommand.empty()) {
    TokenListCIt itEnd = getEndCommand(endCommand, tokList);
    if (itEnd == tokList.end()) {
      printListError(WARN_DOXYGEN_COMMAND_EXPECTED, "Expected Doxygen command: " + endCommand + ".");
      return;
    }
    // If we ignore the command, also ignore any whitespace preceding it as we
    // want to avoid having lines consisting of whitespace only or trailing
    // whitespace in general (at least Python, with its pep8 tool, really
    // doesn't like it).
    if (!doxyList.empty()) {
      DoxygenEntityList::iterator i = doxyList.end();
      --i;
      if (i->typeOfEntity == "plainstd::string" && i->data.find_first_not_of(" \t") == std::string::npos) {
        doxyList.erase(i);
      }
    }
    // Determine what to do with the part of the comment between the start and
    // end commands: by default, we simply throw it away, but "contents"
    // attribute may be used to change this.
    if (String *const contents = getIgnoreFeature(theCommand, "contents")) {
      // Currently only "parse" is supported but we may need to add "copy" to
      // handle custom tags which contain text that is supposed to be copied
      // verbatim in the future.
      if (Strcmp(contents, "parse") == 0) {
        DoxygenEntityList aNewList = parse(itEnd, tokList);
        doxyList.splice(doxyList.end(), aNewList);
      } else {
        Swig_error(m_fileName.c_str(), m_fileLineNo, "Invalid \"doxygen:ignore\" feature \"contents\" attribute \"%s\".\n", Char(contents));
        return;
      }
    }

    m_tokenListIt = itEnd;
    m_tokenListIt++;
  } else if (String *const range = getIgnoreFeature(theCommand, "range")) {
    // Currently we only support "line" but, in principle, we should also
    // support "word" and "paragraph" for consistency with the built-in Doxygen
    // commands which can have either of these three ranges (which are indicated
    // using <word-arg>, (line-arg) and {para-arg} respectively in Doxygen
    // documentation).
    if (Strcmp(range, "line") == 0) {
      // Consume everything until the end of line.
      m_tokenListIt = getOneLine(tokList);
      skipEndOfLine();
    } else {
      Swig_error(m_fileName.c_str(), m_fileLineNo, "Invalid \"doxygen:ignore\" feature \"range\" attribute \"%s\".\n", Char(range));
      return;
    }
  }
}

void DoxygenParser::addCommand(const std::string &commandString, const TokenList &tokList, DoxygenEntityList &doxyList) {

  string theCommand = stringToLower(commandString);

  if (theCommand == "plainstd::string") {
    string nextPhrase = getStringTilCommand(tokList);
    if (noisy)
      cout << "Parsing plain std::string :" << nextPhrase << endl;
    doxyList.push_back(DoxygenEntity("plainstd::string", nextPhrase));
    return;
  }

  switch (commandBelongs(commandString)) {
  case SIMPLECOMMAND:
    addSimpleCommand(theCommand, doxyList);
    break;
  case COMMANDWORD:
    addCommandWord(theCommand, tokList, doxyList);
    break;
  case COMMANDLINE:
    addCommandLine(theCommand, tokList, doxyList);
    break;
  case COMMANDPARAGRAPH:
    addCommandParagraph(theCommand, tokList, doxyList);
    break;
  case COMMANDENDCOMMAND:
    addCommandEndCommand(theCommand, tokList, doxyList);
    break;
  case COMMANDWORDPARAGRAPH:
    addCommandWordParagraph(theCommand, tokList, doxyList);
    break;
  case COMMANDWORDLINE:
    addCommandWordLine(theCommand, tokList, doxyList);
    break;
  case COMMANDWORDOWORDWORD:
    addCommandWordOWordOWord(theCommand, tokList, doxyList);
    break;
  case COMMANDOWORD:
    addCommandOWord(theCommand, tokList, doxyList);
    break;
  case COMMANDERRORTHROW:
    addCommandErrorThrow(theCommand, tokList, doxyList);
    break;
  case COMMANDUNIQUE:
    addCommandUnique(theCommand, tokList, doxyList);
    break;
  case COMMAND_HTML:
    addCommandHtml(theCommand, tokList, doxyList);
    break;
  case COMMAND_HTML_ENTITY:
    addCommandHtmlEntity(theCommand, tokList, doxyList);
    break;
  case COMMAND_ALIAS:
    aliasCommand(commandString, tokList, doxyList);
    break;
  case COMMAND_IGNORE:
    ignoreCommand(commandString, tokList, doxyList);
    break;
  case NONE:
  case END_LINE:
  case PARAGRAPH_END:
  case PLAINSTRING:
  case COMMAND:
    // TODO: Ensure that these values either are correctly ignored here or can't happen.
    break;
  }
}

/**
 * This method converts TokenList to DoxygenEntryList.
 */
DoxygenEntityList DoxygenParser::parse(TokenListCIt endParsingIndex, const TokenList &tokList, bool root) {
  // if we are root, than any strings should be added as 'partofdescription', else as 'plainstd::string'
  std::string currPlainstringCommandType = root ? "partofdescription" : "plainstd::string";
  DoxygenEntityList aNewList;

  // Less than check (instead of not equal) is a safeguard in case the
  // iterator is incremented past the end
  while (m_tokenListIt < endParsingIndex) {

    Token currToken = *m_tokenListIt;

    if (noisy)
      cout << "Parsing for phrase starting in:" << currToken.toString() << endl;

    if (currToken.m_tokenType == END_LINE) {
      aNewList.push_back(DoxygenEntity("plainstd::endl"));
      m_tokenListIt++;
    } else if (currToken.m_tokenType == COMMAND) {
      m_tokenListIt++;
      addCommand(currToken.m_tokenString, tokList, aNewList);
    } else if (currToken.m_tokenType == PLAINSTRING) {
      addCommand(currPlainstringCommandType, tokList, aNewList);
    }

    // If addCommand above misbehaves, it can move the iterator past endParsingIndex
    if (m_tokenListIt > endParsingIndex)
      printListError(WARN_DOXYGEN_UNEXPECTED_ITERATOR_VALUE, "Unexpected iterator value in DoxygenParser::parse");

    if (endParsingIndex != tokList.end() && m_tokenListIt == tokList.end()) {
      // this could happen if we can't reach the original endParsingIndex
      printListError(WARN_DOXYGEN_UNEXPECTED_END_OF_COMMENT, "Unexpected end of Doxygen comment encountered.");
      break;
    }
  }
  return aNewList;
}

DoxygenEntityList DoxygenParser::createTree(Node *node, String *documentation) {
  m_node = node;

  tokenizeDoxygenComment(Char(documentation), Char(Getfile(documentation)), Getline(documentation));

  if (noisy) {
    cout << "---TOKEN LIST---" << endl;
    printList();
  }

  DoxygenEntityList rootList = parse(m_tokenList.end(), m_tokenList, true);

  if (noisy) {
    cout << "PARSED LIST" << endl;
    printTree(rootList);
  }
  return rootList;
}

/*
 * Splits 'text' on 'separator' chars. Separator chars are not part of the
 * strings.
 */
DoxygenParser::StringVector DoxygenParser::split(const std::string &text, char separator) {
  StringVector lines;
  size_t prevPos = 0, pos = 0;

  while (pos < string::npos) {
    pos = text.find(separator, prevPos);
    lines.push_back(text.substr(prevPos, pos - prevPos));
    prevPos = pos + 1;
  }

  return lines;
}

/*
 * Returns true, if 'c' is one of doxygen comment block start
 * characters: *, /, or !
 */
bool DoxygenParser::isStartOfDoxyCommentChar(char c) {
  return (strchr("*/!", c) != NULL);
}

/*
 * Adds token with Doxygen command to token list, but only if command is one of
 * Doxygen commands. In that case true is returned. If the command is not
 * recognized as a doxygen command, it is ignored and false is returned.
 */
bool DoxygenParser::addDoxyCommand(DoxygenParser::TokenList &tokList, const std::string &cmd) {
  if (commandBelongs(cmd) != NONE) {
    tokList.push_back(Token(COMMAND, cmd));
    return true;
  } else {
    if (cmd.empty()) {
      // This actually indicates a bug in the code in this file, as this
      // function shouldn't be called at all in this case.
      printListError(WARN_DOXYGEN_UNKNOWN_COMMAND, "Unexpected empty Doxygen command.");
      return false;
    }

    // This function is called for the special Doxygen commands, but also for
    // HTML commands (or anything that looks like them, actually) and entities.
    // We don't recognize all of those, so just ignore them and pass them
    // through, but warn about unknown Doxygen commands as ignoring them will
    // often result in wrong output being generated.
    const char ch = *cmd.begin();
    if (ch != '<' && ch != '&') {
      // Before calling printListError() we must ensure that m_tokenListIt used
      // by it is valid.
      const TokenListCIt itSave = m_tokenListIt;
      m_tokenListIt = m_tokenList.end();

      printListError(WARN_DOXYGEN_UNKNOWN_COMMAND, "Unknown Doxygen command: " + cmd + ".");

      m_tokenListIt = itSave;
    }
  }

  return false;
}

/*
 * This method copies comment text to output as it is - no processing is
 * done, Doxygen commands are ignored. It is used for commands \verbatim,
 * \htmlonly, \f$, \f[, and \f{.
 */
size_t DoxygenParser::processVerbatimText(size_t pos, const std::string &line) {
  if (line[pos] == '\\' || line[pos] == '@') { // check for end commands

    pos++;
    size_t endOfWordPos = line.find_first_not_of(DOXYGEN_WORD_CHARS, pos);
    string cmd = line.substr(pos, endOfWordPos - pos);

    if (cmd == CMD_END_HTML_ONLY || cmd == CMD_END_VERBATIM || cmd == CMD_END_LATEX_1 || cmd == CMD_END_LATEX_2 || cmd == CMD_END_LATEX_3 || cmd == CMD_END_CODE) {

      m_isVerbatimText = false;
      addDoxyCommand(m_tokenList, cmd);

    } else {

      m_tokenList.push_back(Token(PLAINSTRING,
                                  // include '\' or '@'
                                  line.substr(pos - 1, endOfWordPos - pos + 1)));
    }

    pos = endOfWordPos;

  } else {

    // whitespaces are stored as plain strings
    size_t startOfPossibleEndCmd = line.find_first_of("\\@", pos);
    m_tokenList.push_back(Token(PLAINSTRING, line.substr(pos, startOfPossibleEndCmd - pos)));
    pos = startOfPossibleEndCmd;
  }

  return pos;
}

/*
 * Processes doxy commands for escaped characters: \$ \@ \\ \& \~ \< \> \# \% \" \. \::
 * Handling this separately supports documentation text like \@someText.
 */
bool DoxygenParser::processEscapedChars(size_t &pos, const std::string &line) {
  if ((pos + 1) < line.size()) {

    // \ and @ with trailing whitespace or quoted get to output as plain string
    string whitespaces = " '\t\n";
    if (whitespaces.find(line[pos + 1]) != string::npos) {
      m_tokenList.push_back(Token(PLAINSTRING, line.substr(pos, 1)));
      pos++;
      return true;
    }
    // these chars can be escaped for doxygen
    string escapedChars = "$@\\&~<>#%\".";
    if (escapedChars.find(line[pos + 1]) != string::npos) {

      addDoxyCommand(m_tokenList, line.substr(pos + 1, 1));
      pos += 2;
      return true;

    } else if ((pos + 2) < line.size() && line[pos + 1] == ':' && line[pos + 2] == ':') {

      // add command \:: - handling this separately supports documentation
      // text like \::someText
      addDoxyCommand(m_tokenList, line.substr(pos + 1, 2));
      pos += 3;
      return true;
    }
  }
  return false;
}

/*
 * Processes word doxygen commands, like \arg, \c, \b, \return, ...
 */
void DoxygenParser::processWordCommands(size_t &pos, const std::string &line) {
  pos++;
  size_t endOfWordPos = getEndOfWordCommand(line, pos);

  string cmd = line.substr(pos, endOfWordPos - pos);
  if (cmd.empty()) {
    // This was a bare backslash, just ignore it.
    return;
  }

  addDoxyCommand(m_tokenList, cmd);

  // A flag for whether we want to skip leading spaces after the command
  bool skipLeadingSpace = true;

  if (cmd == CMD_HTML_ONLY || cmd == CMD_VERBATIM || cmd == CMD_LATEX_1 || cmd == CMD_LATEX_2 || cmd == CMD_LATEX_3 || getBaseCommand(cmd) == CMD_CODE) {

    m_isVerbatimText = true;

    // Skipping leading space is necessary with inline \code command,
    // and it won't hurt anything for block \code (TODO: are the other
    // commands also compatible with skip leading space?  If so, just
    // do it every time.)
    if (getBaseCommand(cmd) == CMD_CODE) skipLeadingSpace = true;
    else skipLeadingSpace = false;
  } else if (cmd.substr(0,3) == "end") {
    // If processing an "end" command such as "endlink", don't skip
    // the space before the next string
    skipLeadingSpace = false;
  }

  if (skipLeadingSpace) {
    // skip any possible spaces after command, because some commands have parameters,
    // and spaces between command and parameter must be ignored.
    if (endOfWordPos != string::npos) {
      endOfWordPos = line.find_first_not_of(" \t", endOfWordPos);
    }
  }
  
  pos = endOfWordPos;
}

void DoxygenParser::processHtmlTags(size_t &pos, const std::string &line) {
  bool isEndHtmlTag = false;
  pos++;
  if (line.size() > pos && line[pos] == '/') {
    isEndHtmlTag = true;
    pos++;
  }

  size_t endHtmlPos = line.find_first_of("\t >", pos);

  string cmd = line.substr(pos, endHtmlPos - pos);
  pos = endHtmlPos;

  // prepend '<' to distinguish HTML tags from doxygen commands
  if (!cmd.empty() && addDoxyCommand(m_tokenList, '<' + cmd)) {
    // it is a valid HTML command
    if (pos == string::npos) {
      pos = line.size();
    }
    if (line[pos] != '>') {
      // it should be HTML tag with args,
      // for example <A ...>, <IMG ...>, ...
      if (isEndHtmlTag) {
        m_tokenListIt = m_tokenList.end();
        printListError(WARN_DOXYGEN_HTML_ERROR, "Doxygen HTML error for tag " + cmd + ": Illegal end HTML tag without greater-than ('>') found.");
      }

      endHtmlPos = line.find('>', pos);
      if (endHtmlPos == string::npos) {
        m_tokenListIt = m_tokenList.end();
        printListError(WARN_DOXYGEN_HTML_ERROR, "Doxygen HTML error for tag " + cmd + ": HTML tag without greater-than ('>') found.");
      }
      // add args of HTML command, like link URL, image URL, ...
      m_tokenList.push_back(Token(PLAINSTRING, line.substr(pos, endHtmlPos - pos)));
      pos = endHtmlPos;
    } else {
      if (isEndHtmlTag) {
        m_tokenList.push_back(Token(PLAINSTRING, END_HTML_TAG_MARK));
      } else {
        // it is a simple tag, so push empty string
        m_tokenList.push_back(Token(PLAINSTRING, ""));
      }
    }

    if (pos < line.size()) {
      pos++; // skip '>'
    }
  } else {
    // the command is not HTML supported by Doxygen, < and > will be
    // replaced by HTML entities &lt; and &gt; respectively,
    addDoxyCommand(m_tokenList, "&lt");
    m_tokenList.push_back(Token(PLAINSTRING, cmd));
  }
}

void DoxygenParser::processHtmlEntities(size_t &pos, const std::string &line) {
  size_t endOfWordPos = line.find_first_not_of("abcdefghijklmnopqrstuvwxyz", pos + 1);

  if (endOfWordPos != string::npos) {

    if (line[endOfWordPos] == ';' && (endOfWordPos - pos) > 1) {
      // if entity is not recognized by Doxygen (not in the list of
      // commands) nothing is added (here and in Doxygen).
      addDoxyCommand(m_tokenList, line.substr(pos, endOfWordPos - pos));
      endOfWordPos++; // skip ';'
    } else {
      // it is not an entity - add entity for ampersand and the rest of string
      addDoxyCommand(m_tokenList, "&amp");
      m_tokenList.push_back(Token(PLAINSTRING, line.substr(pos + 1, endOfWordPos - pos - 1)));
    }
  }
  pos = endOfWordPos;
}

/*
 * This method processes normal comment, which has to be tokenized.
 */
size_t DoxygenParser::processNormalComment(size_t pos, const std::string &line) {
  switch (line[pos]) {
  case '\\':
  case '@':
    if (processEscapedChars(pos, line)) {
      break;
    }
    // handle word commands \arg, \c, \return, ... and \f[, \f$, ... commands
    processWordCommands(pos, line);
    break;

  case ' ': // whitespace
  case '\t':
    {
      // whitespaces are stored as plain strings
      size_t startOfNextWordPos = line.find_first_not_of(" \t", pos + 1);
      m_tokenList.push_back(Token(PLAINSTRING, line.substr(pos, startOfNextWordPos - pos)));
      pos = startOfNextWordPos;
    }
    break;

  case '<':
    processHtmlTags(pos, line);
    break;
  case '>': // this char is detected here only when it is not part of HTML tag
    addDoxyCommand(m_tokenList, "&gt");
    pos++;
    break;
  case '&':
    processHtmlEntities(pos, line);
    break;
  case '"':
    m_isInQuotedString = true;
    m_tokenList.push_back(Token(PLAINSTRING, "\""));
    pos++;
    break;
  default:
    m_tokenListIt = m_tokenList.end();
    printListError(WARN_DOXYGEN_UNKNOWN_CHARACTER, std::string("Unknown special character in Doxygen comment: ") + line[pos] + ".");
  }

  return pos;
}

/*
 * This is the main method, which tokenizes Doxygen comment to words and
 * doxygen commands.
 */
void DoxygenParser::tokenizeDoxygenComment(const std::string &doxygenComment, const std::string &fileName, int fileLine) {
  m_isVerbatimText = false;
  m_isInQuotedString = false;
  m_tokenList.clear();
  m_fileLineNo = fileLine;
  m_fileName = fileName;

  StringVector lines = split(doxygenComment, '\n');

  // remove trailing spaces, because they cause additional new line at the end
  // comment, which is wrong, because these spaces are space preceding
  // end of comment :  '  */'
  if (!doxygenComment.empty() && doxygenComment[doxygenComment.size() - 1] == ' ') {

    string lastLine = lines[lines.size() - 1];

    if (trim(lastLine).empty()) {
      lines.pop_back(); // remove trailing empty line
    }
  }

  for (StringVectorCIt it = lines.begin(); it != lines.end(); it++) {
    const string &line = *it;
    size_t pos = line.find_first_not_of(" \t");

    if (pos == string::npos) {
      m_tokenList.push_back(Token(END_LINE, "\n"));
      continue;
    }
    // skip sequences of '*', '/', and '!' of any length
    bool isStartOfCommentLineCharFound = false;
    while (pos < line.size() && isStartOfDoxyCommentChar(line[pos])) {
      pos++;
      isStartOfCommentLineCharFound = true;
    }

    if (pos == line.size()) {
      m_tokenList.push_back(Token(END_LINE, "\n"));
      continue;
    }
    // if 'isStartOfCommentLineCharFound' then preserve leading spaces, so
    // ' *    comment' gets translated to ' *    comment', not ' * comment'
    // This is important to keep formatting for comments translated to Python.
    if (isStartOfCommentLineCharFound && line[pos] == ' ') {
      pos++; // points to char after ' * '
      if (pos == line.size()) {
        m_tokenList.push_back(Token(END_LINE, "\n"));
        continue;
      }
    }
    // line[pos] may be ' \t' or start of word, it there was no '*', '/' or '!'
    // at beginning of the line. Make sure it points to start of the first word
    // in the line.
    if (isStartOfCommentLineCharFound) {
      size_t firstWordPos = line.find_first_not_of(" \t", pos);
      if (firstWordPos == string::npos) {
        m_tokenList.push_back(Token(END_LINE, "\n"));
        continue;
      }

      if (firstWordPos > pos) {
        m_tokenList.push_back(Token(PLAINSTRING, line.substr(pos, firstWordPos - pos)));
        pos = firstWordPos;
      }
    } else {
      m_tokenList.push_back(Token(PLAINSTRING, line.substr(0, pos)));
    }

    while (pos != string::npos) {
      // find the end of the word
      size_t doxyCmdOrHtmlTagPos = line.find_first_of("\\@<>&\" \t", pos);
      if (doxyCmdOrHtmlTagPos != pos) {
        // plain text found
        // if the last char is punctuation, make it a separate word, otherwise
        // it may be included with word also when not appropriate, for example:
        //   colors are \b red, green, and blue --> colors are <b>red,</b> green, and blue
        // instead of (comma not bold):
        //   colors are \b red, green, and blue --> colors are <b>red</b>, green, and blue
        // In Python it looks even worse:
        //   colors are \b red, green, and blue --> colors are 'red,' green, and blue
        string text = line.substr(pos, doxyCmdOrHtmlTagPos - pos);
        string punctuations(".,:");
        size_t textSize = text.size();

        if (!text.empty()
            && punctuations.find(text[text.size() - 1]) != string::npos &&
            // but do not break ellipsis (...)
            !(textSize > 1 && text[textSize - 2] == '.')) {
          m_tokenList.push_back(Token(PLAINSTRING, text.substr(0, text.size() - 1)));
          m_tokenList.push_back(Token(PLAINSTRING, text.substr(text.size() - 1)));
        } else {
          m_tokenList.push_back(Token(PLAINSTRING, text));
        }
      }

      pos = doxyCmdOrHtmlTagPos;
      if (pos != string::npos) {
        if (m_isVerbatimText) {
          pos = processVerbatimText(pos, line);

        } else if (m_isInQuotedString) {

          if (line[pos] == '"') {
            m_isInQuotedString = false;
          }
          m_tokenList.push_back(Token(PLAINSTRING, line.substr(pos, 1)));
          pos++;

        } else {
          pos = processNormalComment(pos, line);
        }
      }
    }
    m_tokenList.push_back(Token(END_LINE, "\n")); // add when pos == npos - end of line
  }

  m_tokenListIt = m_tokenList.begin();
}

void DoxygenParser::printList() {

  int tokNo = 0;
  for (TokenListCIt it = m_tokenList.begin(); it != m_tokenList.end(); it++, tokNo++) {

    cout << it->toString() << ' ';

    if ((tokNo % TOKENSPERLINE) == 0) {
      cout << endl;
    }
  }
}

void DoxygenParser::printListError(int warningType, const std::string &message) {
  int curLine = m_fileLineNo;
  for (TokenListCIt it = m_tokenList.begin(); it != m_tokenListIt; it++) {
    if (it->m_tokenType == END_LINE) {
      curLine++;
    }
  }

  Swig_warning(warningType, m_fileName.c_str(), curLine, "%s\n", message.c_str());
}
