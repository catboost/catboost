%%{
    machine CharacterNames;

    yc_sp = yc_09 | yc_0A | yc_0B | yc_0C | yc_0D | yc_20; # [\t\n\v\f\r ]

    EOF = 0; # end of stream marker (we'll remove this requirement with advance to rl6's eof)

    # character synonyms
    yc_zero          = 0x0000; # '\0';
    yc_tab           = 0x0009; # '\t';

    yc_space         = 0x0020; # ' ';
    yc_exclamation   = 0x0021; # '!';
    yc_quotation     = yc_22; # '"';
    yc_number_sign   = 0x0023; # '#';
    yc_dollar        = 0x0024; # '$';
    yc_percent       = 0x0025; # '%';
    yc_ampersand     = 0x0026; # '&';
    yc_apostrophe    = yc_27; # '\'';
    yc_left_paren    = 0x0028; # '(';
    yc_right_paren   = 0x0029; # ')';
    yc_asterisk      = 0x002A; # '*';
    yc_plus          = 0x002B; # '+';
    yc_comma         = 0x002C; # ',';
    yc_minus         = 0x002D; # '-';
    yc_dot           = 0x002E; # '.';
    yc_slash         = 0x002F; # '/';

    yc_colon         = 0x003A; # ':';
    yc_less          = 0x003C; # '<';
    yc_equals        = 0x003D; # '=';
    yc_greater       = 0x003E; # '>';

    yc_at_sign       = 0x0040; # '@';

    yc_left_bracket  = 0x005B; # '[';
    #yc_backslash    = 0x005C; # '\\';
    yc_right_bracket = 0x005D; # ']';
    yc_caret         = 0x005E; # '^';
    yc_underscore    = 0x005F; # '_';

    yc_accent        = yc_60; # '`';

    yc_left_brace    = 0x007B; # '{';
    yc_vert_bar      = 0x007C; # '|';
    yc_right_brace   = 0x007D; # '}';
    yc_tilde         = 0x007E; # '~';

    # old character synonyms
    yc_wide_zero          = 0x00;  # '\0';
    yc_wide_tab           = yc_09; # '\t';

    yc_wide_space         = yc_20; # ' ';
    yc_wide_exclamation   = yc_21; # '!';
    yc_wide_quotation     = yc_22; # '"';
    yc_wide_number_sign   = yc_23; # '#';
    yc_wide_dollar        = yc_24; # '$';
    yc_wide_percent       = yc_25; # '%';
    yc_wide_ampersand     = yc_26; # '&';
    yc_wide_apostrophe    = yc_27; # '\'';
    yc_wide_left_paren    = yc_28; # '(';
    yc_wide_right_paren   = yc_29; # ')';
    yc_wide_asterisk      = yc_2A; # '*';
    yc_wide_plus          = yc_2B; # '+';
    yc_wide_comma         = yc_2C; # ',';
    yc_wide_minus         = yc_2D; # '-';
    yc_wide_dot           = yc_2E; # '.';
    yc_wide_slash         = yc_2F; # '/';

    yc_wide_colon         = yc_3A; # ':';
    yc_wide_less          = yc_3C; # '<';
    yc_wide_equals        = yc_3D; # '=';
    yc_wide_greater       = yc_3E; # '>';

    yc_wide_at_sign       = yc_40; # '@';

    yc_wide_left_bracket  = yc_5B; # '[';
    #yc_wide_backslash    = yc_5C; # '\\';
    yc_wide_right_bracket = yc_5D; # ']';
    yc_wide_caret         = yc_5E; # '^';
    yc_wide_underscore    = yc_5F; # '_';

    yc_wide_accent        = yc_60; # '`';

    yc_wide_left_brace    = yc_7B; # '{';
    yc_wide_vert_bar      = yc_7C; # '|';
    yc_wide_right_brace   = yc_7D; # '}';
    yc_wide_tilde         = yc_7E; # '~';

}%%
