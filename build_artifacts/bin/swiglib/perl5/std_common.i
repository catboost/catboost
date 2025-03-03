/* -----------------------------------------------------------------------------
 * std_common.i
 *
 * SWIG typemaps for STL - common utilities
 * ----------------------------------------------------------------------------- */

%include <std/std_except.i>

%apply size_t { std::size_t };

%fragment("<string>");
%{
SWIGINTERN
double SwigSvToNumber(SV* sv) {
    return SvIOK(sv) ? double(SvIVX(sv)) : SvNVX(sv);
}
SWIGINTERN
std::string SwigSvToString(SV* sv) {
    STRLEN len;
    char *ptr = SvPV(sv, len);
    return std::string(ptr, len);
}
SWIGINTERN
void SwigSvFromString(SV* sv, const std::string& s) {
    sv_setpvn(sv,s.data(),s.size());
}
%}

