%%{
    machine MultitokenDef;

    # AddLastToken(ts, tokend) should be implemented except member functions called here

    action begin_token {
        BeginToken(ts, p);
    }

    action begin_word {
        BeginToken(ts, p, TOKEN_WORD);
    }

    action begin_number {
        BeginToken(ts, p, TOKEN_NUMBER);
    }

    action update_token {
        UpdateToken();
    }

    action add_token {
        AddToken();
    }

    action update_prefix {
        UpdatePrefix(*p);
    }

    action update_suffix {
        UpdateSuffix(*p);
    }

    # @ATTENTION if '%' is added to subtokdelim it breaks the code in MakeMultitokenEntry(): utf8 = Find(.., PERCENT_CHAR, ..);
    #            in this case two chars that follow '%' must be checked for one of '0123456789ABCDEF'
    # @note when '%' action fired 'p' points to the next character so to take the previous character use 'p[-1]'

    tokendelim = ( cc_apostrophe %{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); } )
               | ( cc_minus      %{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); } );     # ['-] = tokdelim

    multitokendelim = ( cc_plus       %{ SetTokenDelim(TOKDELIM_PLUS, p[-1]); } )
                    | ( cc_underscore %{ SetTokenDelim(TOKDELIM_UNDERSCORE, p[-1]); } )
                    | ( cc_slash      %{ SetTokenDelim(TOKDELIM_SLASH, p[-1]); } )
                    | ( cc_atsign     %{ SetTokenDelim(TOKDELIM_AT_SIGN, p[-1]); } )
                    | ( cc_dot        %{ SetTokenDelim(TOKDELIM_DOT, p[-1]); } );      # [+_/@.] = identdelim + [.]

    tokpart = ( tokchar ( tokchar | accent )* ); # | ( yspecialkey );
    numpart = ( ydigit ( ydigit | accent )* );

    tokfirst = ( ( ( accent* >begin_token ) ( tokpart >begin_word ) ) $update_token %add_token );
    tokfirst_special = ( ( ( accent* >begin_token ) ( yspecialkey >begin_word ) ) $update_token %add_token );
    toknext  = (                              tokpart >begin_word     $update_token %add_token );

    numfirst = ( ( ( accent* >begin_token ) ( numpart >begin_number ) ) $update_token %add_token );
    numnext  = (                              numpart >begin_number     $update_token %add_token );

    #wordpart = tokfirst;

    toksuffix = (cc_numbersign | cc_plus | cc_plus.cc_plus) $update_suffix; # ([#] | [+] | [+][+])

    # - in case of " abc&x301;123 " accent is attached to "abc"
    # - 'accent*' cannot be removed from the front 'token' and 'number' because in this case text "abc-&x301;123" or
    #   "123-&x301;abc" it will be processed incorrectly
    # - begin_token can be called twice in case "exa&shy;&#x301;mple" so BeginToken() has 'if (CurCharSpan.Len == 0)'
    #   and it processes only the first call

    solidtoken = (         tokfirst ( numnext toknext )* )
               | ( numfirst toknext ( numnext toknext )* )
               | (         numfirst ( toknext numnext )* )
               | ( tokfirst numnext ( toknext numnext )* )
               | (tokfirst_special);

    multitoken = ( solidtoken ( tokendelim solidtoken ){,4} );
    multitokenwithsuffix = ( ( tokprefix $update_prefix )?  multitoken toksuffix? );
    compositemultitoken = ( multitokenwithsuffix ( multitokendelim multitokenwithsuffix )* );

}%%
