Some usage notes for the D Parser:

- it is a port of the Java parser, so interface is very similar.

- the lexer class needs to implement the interface 'Lexer' (similar to
  java). It typically (depending on options) looks like this:

public interface Lexer
{
  /**
   * Method to retrieve the beginning position of the last scanned token.
   * @return the position at which the last scanned token starts.  */
  @property YYPosition startPos ();

  /**
   * Method to retrieve the ending position of the last scanned token.
   * @return the first position beyond the last scanned token.  */
  @property YYPosition endPos ();

  /**
   * Method to retrieve the semantic value of the last scanned token.
   * @return the semantic value of the last scanned token.  */
  @property YYSemanticType semanticVal ();

  /**
   * Entry point for the scanner.  Returns the token identifier corresponding
   * to the next token and prepares to return the semantic value
   * and beginning/ending positions of the token.
   * @return the token identifier corresponding to the next token. */
  TokenKind yylex ();

  /**
   * Entry point for error reporting.  Emits an error
   * referring to the given location in a user-defined way.
   *
   * @param loc The location of the element to which the
   *                error message is related
   * @param s The string for the error message.  */
   void yyerror (YYLocation loc, string s);
}

- semantic types are handled by D unions (same as for C/C++ parsers)

- the following (non-standard) %defines are supported:

  %define package "<package_name>"
  %define api.parser.class "my_class_name>"
  %define position_type "my_position_type"
  %define location_type "my_location_type"

- the following declarations basically work like in C/C++:

  %locations
  %error-verbose
  %parse-param
  %initial-action
  %code
  %union

- %destructor is not yet supported
