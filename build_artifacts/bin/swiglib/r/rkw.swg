/*
  Warnings for R keywords, built-in names and bad names.
*/

#define RKW(x) %keywordwarn("'" `x` "' is a R keyword, renaming to '_" `x`"'", rename="_%s")  `x`
#define RSWIGKW(x) %keywordwarn("'" `x` "' is a SWIG R reserved parameter name, renaming to '_" `x`"'", rename="_%s")  `x`

/*
  Warnings for R reserved words taken from
  http://cran.r-project.org/doc/manuals/R-lang.html#Reserved-words
*/

RKW(if);
RKW(else);
RKW(repeat);
RKW(while);
RKW(function);
RKW(for);
RKW(in);
RKW(next);
RKW(break);
RKW(TRUE);
RKW(FALSE);
RKW(NULL);
RKW(Inf);
RKW(NaN);
RKW(NA);
RKW(NA_integer_);
RKW(NA_real_);
RKW(NA_complex_);
RKW(NA_character_);

RSWIGKW(self);

#undef RKW
#undef RSWIGKW
