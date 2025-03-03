#ifndef OCAML_OCAMLKW_SWG_
#define OCAML_OCAMLKW_SWG_

/* Warnings for Ocaml keywords */
#define OCAMLKW(x) %namewarn("314: '" #x "' is an ocaml keyword and it will be appropriately renamed")  #x

/*
  from
  https://caml.inria.fr/pub/docs/manual-ocaml/lex.html
*/


OCAMLKW(and);
OCAMLKW(as);
OCAMLKW(asr);
OCAMLKW(assert);
OCAMLKW(begin);
OCAMLKW(class);
OCAMLKW(constraint);
OCAMLKW(do);
OCAMLKW(done);
OCAMLKW(downto);
OCAMLKW(else);
OCAMLKW(end);
OCAMLKW(exception);
OCAMLKW(external);
OCAMLKW(false);
OCAMLKW(for);
OCAMLKW(fun);
OCAMLKW(function);
OCAMLKW(functor);
OCAMLKW(if);
OCAMLKW(in);
OCAMLKW(include);
OCAMLKW(inherit);
OCAMLKW(initializer);
OCAMLKW(land);
OCAMLKW(lazy);
OCAMLKW(let);
OCAMLKW(lor);
OCAMLKW(lsl);
OCAMLKW(lsr);
OCAMLKW(lxor);
OCAMLKW(match);
OCAMLKW(method);
OCAMLKW(mod);
OCAMLKW(module);
OCAMLKW(mutable);
OCAMLKW(new);
OCAMLKW(nonrec);
OCAMLKW(object);
OCAMLKW(of);
OCAMLKW(open);
OCAMLKW(or);
OCAMLKW(private);
OCAMLKW(rec);
OCAMLKW(sig);
OCAMLKW(struct);
OCAMLKW(then);
OCAMLKW(to);
OCAMLKW(true);
OCAMLKW(try);
OCAMLKW(type);
OCAMLKW(val);
OCAMLKW(virtual);
OCAMLKW(when);
OCAMLKW(while);
OCAMLKW(with);

#undef OCAMLKW

#endif //OCAML_OCAMLKW_SWG_
