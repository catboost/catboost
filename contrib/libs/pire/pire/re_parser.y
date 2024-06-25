%{ // -*- mode: c++ -*-

/*
 * re_parser.ypp -- the main regexp parsing routine
 *
 * Copyright (c) 2007-2010, Dmitry Prokoptsev <dprokoptsev@gmail.com>,
 *                          Alexander Gololobov <agololobov@gmail.com>
 *
 * This file is part of Pire, the Perl Incompatible
 * Regular Expressions library.
 *
 * Pire is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Pire is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser Public License for more details.
 * You should have received a copy of the GNU Lesser Public License
 * along with Pire.  If not, see <http://www.gnu.org/licenses>.
 */


#ifdef _MSC_VER
// Disable yacc warnings
#pragma warning(disable: 4060) // switch contains no 'case' or 'default' statements
#pragma warning(disable: 4065) // switch contains 'default' but no 'case' statements
#pragma warning(disable: 4102) // unreferenced label 'yyerrlabl'
#pragma warning(disable: 4702) // unreachable code
#endif

#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wuninitialized" // 'yylval' may be used uninitialized
#endif

#include <stdexcept>

#include <contrib/libs/pire/pire/fsm.h>
#include <contrib/libs/pire/pire/re_lexer.h>
#include <contrib/libs/pire/pire/any.h>
#include <contrib/libs/pire/pire/stub/stl.h>

#define YYSTYPE Any*
#define YYSTYPE_IS_TRIVIAL 0

namespace {

using namespace Pire;
using Pire::Fsm;
using Pire::Encoding;

int  yylex(YYSTYPE*, Lexer&);
void yyerror(Pire::Lexer&, const char*);

Fsm& ConvertToFSM(const Encoding& encoding, Any* any);
void AppendRange(const Encoding& encoding, Fsm& a, const Term::CharacterRange& cr);

%}

%parse-param { Pire::Lexer& rlex }
%lex-param { Pire::Lexer& rlex }
%pure-parser

// Terminal declarations
%term YRE_LETTERS
%term YRE_COUNT
%term YRE_DOT
%term YRE_AND
%term YRE_NOT

%destructor { delete $$; } <>

%%

regexp
    : alternative
        {
            ConvertToFSM(rlex.Encoding(), $1);
            DoSwap(rlex.Retval(), *$1);
            delete $1;
            $$ = nullptr;
        }
    ;

alternative
    : conjunction
    | alternative '|' conjunction { ConvertToFSM(rlex.Encoding(), ($$ = $1)) |= ConvertToFSM(rlex.Encoding(), $3); delete $2; delete $3; }
    ;

conjunction
    : negation
    | conjunction YRE_AND negation { ConvertToFSM(rlex.Encoding(), ($$ = $1)) &= ConvertToFSM(rlex.Encoding(), $3); delete $2; delete $3; }
    ;

negation
    : concatenation
    | YRE_NOT concatenation { ConvertToFSM(rlex.Encoding(), ($$ = $2)).Complement(); delete $1; }
    ;

concatenation
    : { $$ = new Any(Fsm()); }
    | concatenation iteration
        {
            Fsm& a = ConvertToFSM(rlex.Encoding(), ($$ = $1));
            if ($2->IsA<Term::CharacterRange>() && !$2->As<Term::CharacterRange>().second)
                AppendRange(rlex.Encoding(), a, $2->As<Term::CharacterRange>());
            else if ($2->IsA<Term::DotTag>())
                rlex.Encoding().AppendDot(a);
            else
                a += ConvertToFSM(rlex.Encoding(), $2);
            delete $2;
        }
    ;

iteration
    : term
    | term YRE_COUNT
        {
            Fsm& orig = ConvertToFSM(rlex.Encoding(), $1);
            $$ = new Any(orig);
            Fsm& cur = $$->As<Fsm>();
            const Term::RepetitionCount& repc = $2->As<Term::RepetitionCount>();


            if (repc.first == 0 && repc.second == 1) {
                Fsm empty;
                cur |= empty;
            } else if (repc.first == 0 && repc.second == Inf) {
                cur.Iterate();
            } else if (repc.first == 1 && repc.second == Inf) {
                cur += *cur;
            } else {
                cur *= repc.first;
                if (repc.second == Inf) {
                    cur += *orig;
                } else if (repc.second != repc.first) {
                    cur += (orig | Fsm()) * (repc.second - repc.first);
                }
            }
            rlex.Parenthesized($$->As<Fsm>());
            delete $1;
            delete $2;
        }
    ;

term
    : YRE_LETTERS
    | YRE_DOT
    | '^'
    | '$'
    | '(' alternative ')'      { $$ = $2; rlex.Parenthesized($$->As<Fsm>()); delete $1; delete $3; }
    ;

%%

int yylex(YYSTYPE* lval, Pire::Lexer& rlex)
{
    try {
        Pire::Term term = rlex.Lex();
        if (!term.Value().Empty())
            *lval = new Any(term.Value());
        else
            *lval = nullptr;
        return term.Type();
    } catch (Pire::Error &e) {
        rlex.SetError(e.what());
        return 0;
    }
}

void yyerror(Pire::Lexer& rlex, const char* str)
{
    if (rlex.GetError().length() == 0)
        rlex.SetError(ystring("Regexp parse error: ").append(str));
}

void AppendRange(const Encoding& encoding, Fsm& a, const Term::CharacterRange& cr)
{
    TVector<ystring> strings;

    for (auto&& i : cr.first) {
        ystring s;
        for (auto&& j : i) {
            ystring c = encoding.ToLocal(j);
            if (c.empty()) {
                s.clear();
                break;
            } else
                s += encoding.ToLocal(j);
        }
        if (!s.empty())
            strings.push_back(s);
    }
    if (strings.empty())
        // Strings accepted by this FSM are not representable in the current encoding.
        // Hence, FSM will accept nothing, and we simply can clear it.
        a = Fsm::MakeFalse();
    else
        a.AppendStrings(strings);
}

Fsm& ConvertToFSM(const Encoding& encoding, Any* any)
{
    if (any->IsA<Fsm>())
        return any->As<Fsm>();

    Any ret = Fsm();
    Fsm& a = ret.As<Fsm>();

    if (any->IsA<Term::DotTag>()) {
        encoding.AppendDot(a);
    } else if (any->IsA<Term::BeginTag>()) {
        a.AppendSpecial(BeginMark);
    } else if (any->IsA<Term::EndTag>()) {
        a.AppendSpecial(EndMark);
    } else {
        Term::CharacterRange cr = any->As<Term::CharacterRange>();
        AppendRange(encoding, a, cr);
        if (cr.second) {
            Fsm x;
            encoding.AppendDot(x);
            x.Complement();
            a |= x;
            a.Complement();
            a.RemoveDeadEnds();
        }
    }
    any->Swap(ret);
    return a;
}

}

namespace Pire {
    namespace Impl {
        int yre_parse(Pire::Lexer& rlex)
        {
            int rc = yyparse(rlex);

            if (rlex.GetError().length() != 0)
                throw Error(rlex.GetError());
            return rc;
        }
    }
}
