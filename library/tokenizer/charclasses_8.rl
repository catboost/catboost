%%{

machine CharacterClasses;
alphtype unsigned char;

#############################################
# Named Characters

cc_zero = 0x00; # (EOF) [\0]
cc_tab = 0x09; # [\t]
cc_linefeed = 0x0A; # [\n]
cc_carriagereturn = 0x0D; # [\r]
cc_space = 0x20; # [ ]
cc_quotationmark = 0x22; # ["]
cc_numbersign = 0x23; # [#]
cc_dollarsign = 0x24; # [$]
cc_percent = 0x25; # [%]
cc_ampersand = 0x26; # [&]
cc_apostrophe = 0x27; # [']
cc_asterisk = 0x2A; # [*]
cc_plus = 0x2B; # [+]
cc_comma = 0x2C; # [,]
cc_minus = 0x2D; # [-]
cc_dot   = 0x2E; # [.]
cc_slash = 0x2F; # [/]
cc_digit = 0x31; # [1]
cc_atsign = 0x40; # [@]
cc_capitalalpha = 0x41; # [A]
cc_underscore = 0x5F; # [_]
cc_smallalpha = 0x61; # [a]
cc_accent = 0x80;
cc_unicasealpha = 0x81; # georgian, hebrew, arabic alphabets
cc_softhyphen = 0x8F;
cc_ideograph = 0x9F;
cc_nbsp = 0xA0;
cc_sectionsign = 0xA7;
cc_copyrightsign = 0xA9;
cc_special = 0xB0;

cc_math = 0xC0;
cc_math_non_ascii = 0xD0;
cc_currency_non_ascii = 0xD1;
cc_special_non_ascii = 0xD2;

#############################################
# Classes

# = 0xB1;
cc_openpunct = 0xB2 | # [(\[{]
    cc_apostrophe | cc_quotationmark; # opening punctuation
cc_clospunct = 0xB3 | # [)\]}]
    cc_apostrophe | cc_quotationmark; # closing punctuation
cc_surrogatelead = 0xB4;
cc_surrogatetail = 0xB5;
cc_whitespace = 0xB6 | cc_tab | cc_linefeed | cc_carriagereturn | cc_space; # [\t\n\v\f\r ]
cc_numerosign = 0xB7; # unicode 0x2116
# = 0xB8;
# = 0xB9;
cc_cjk_termpunct = 0xBA; # fullwidth cjk terminating punctuation
cc_termpunct = 0xBB | cc_dot; # terminating punctuation [!.?] | [!.;?]
cc_currency = cc_dollarsign | cc_currency_non_ascii;
cc_control = 0xBD | # 0x01 - 0x1F, 0x7F excluding
    cc_tab | cc_linefeed | cc_carriagereturn;
cc_misctext = 0xBE | cc_math | cc_math_non_ascii | # [:;<=>\^`|~]
    cc_control | cc_whitespace | cc_comma | cc_asterisk | cc_ampersand |
    cc_termpunct | cc_openpunct | cc_clospunct | cc_numbersign | cc_currency | cc_percent |
    cc_plus | cc_minus | cc_dot | cc_slash | cc_atsign | cc_underscore;

cc_unknown = 0xFF;

}%%

