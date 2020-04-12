%%{
    machine Symbols;

    include CharacterClasses "charclasses_8.rl";

    #
    # CODES_YANDEX symbols
    #

    EOF = cc_zero;
    accent = cc_accent; # required for multitoken.rl

    yc_lf = cc_linefeed; # [\n]
    yc_cr = cc_carriagereturn; # [\r]
    yc_sp = cc_whitespace; # [\t\n\v\f\r ]

    yspecialkey = cc_math_non_ascii | cc_currency_non_ascii | cc_special_non_ascii | cc_numerosign | cc_copyrightsign;
    yspecial = accent | cc_softhyphen | cc_nbsp | cc_sectionsign | cc_special | cc_special_non_ascii | cc_numerosign | cc_copyrightsign;

    ydigit = cc_digit;
    ycapital = cc_capitalalpha;
    ysmall = cc_smallalpha;

    yalpha = ycapital | ysmall | cc_unicasealpha;
    yalnum = ydigit | yalpha;

    ytitle = ydigit | ycapital | cc_unicasealpha; # may be at the beginning of sentence
    cjk_title = ytitle | cc_ideograph;
    ylower = ysmall;            # the same as (yalnum - ytitle)

    termpunct = cc_termpunct;
    cjk_termpunct = cc_cjk_termpunct;

    # Multitoken composition: delimiters and suffixes
    tokdelim = cc_apostrophe | cc_minus;   # [\'\-] TODO: add yc_underscore [_]
    tokprefix = cc_numbersign | cc_atsign | cc_dollarsign; # [#@$]

    # 1..31 | termpunct | [ \"#\$%&\'()*+,\-/;<=>@\[\\\]\^_\`{|}~] | 0x7F | yspecial
    # yc_07 and yc_1B do not exist
    miscnlp =
        (cc_nbsp  |
        cc_misctext |
        yspecial) - yspecialkey;

    # fallback
    othermisc = any - yalnum - cc_zero - miscnlp - cc_ideograph - cc_surrogatelead - cc_surrogatetail - yspecialkey;
}%%

