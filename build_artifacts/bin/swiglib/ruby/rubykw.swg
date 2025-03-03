#ifndef RUBY_RUBYKW_SWG_
#define RUBY_RUBYKW_SWG_

/* Warnings for Ruby keywords */
#define RUBYKW(x) %keywordwarn("'" `x` "' is a ruby keyword, renaming to 'C_" `x` "'",rename="C_%s",fullname=1)  `x`

/*

   from http://www.rubycentral.com/book/language.html

*/

RUBYKW(BEGIN);
RUBYKW(END);
RUBYKW(alias);
RUBYKW(and);
RUBYKW(begin);
RUBYKW(break);
RUBYKW(case);
RUBYKW(class);
RUBYKW(def);
RUBYKW("defined");
RUBYKW(do);
RUBYKW(else);
RUBYKW(elsif);
RUBYKW(end);
RUBYKW(ensure);
RUBYKW(false);
RUBYKW(fatal);
RUBYKW(for);
RUBYKW(if);
RUBYKW(in);
RUBYKW(module);
RUBYKW(next);
RUBYKW(nil);
RUBYKW(not);
RUBYKW(or);
RUBYKW(redo);
RUBYKW(rescue);
RUBYKW(retry);
RUBYKW(return);
RUBYKW(self);
RUBYKW(super);
RUBYKW(then);
RUBYKW(true);
RUBYKW(undef);
RUBYKW(unless);
RUBYKW(until);
RUBYKW(when);
RUBYKW(while);
RUBYKW(yield);

// RUBYKW(FalseClass);
// RUBYKW(TrueClass);
// RUBYKW(Numeric);
// RUBYKW(Integer);
// RUBYKW(Fixnum);
// RUBYKW(Float);
// RUBYKW(Range);
// RUBYKW(Array);
// RUBYKW(String);
// RUBYKW(IO);
// RUBYKW(File);
// RUBYKW(FileUtils);
// RUBYKW(Find);
// RUBYKW(Struct);
// RUBYKW(OpenStruct);
// RUBYKW(Regexp);

#undef RUBYKW

#endif //RUBY_RUBYKW_SWG_
