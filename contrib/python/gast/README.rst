GAST, daou naer!
================

A generic AST to represent Python2 and Python3's Abstract Syntax Tree(AST).

GAST provides a compatibility layer between the AST of various Python versions,
as produced by ``ast.parse`` from the standard ``ast`` module.

Basic Usage
-----------

.. code:: python

    >>> import ast, gast
    >>> code = open('file.py').read()
    >>> tree = ast.parse(code)
    >>> gtree = gast.ast_to_gast(tree)
    >>> ... # process gtree
    >>> tree = gast.gast_to_ast(gtree)
    >>> ... # do stuff specific to tree

API
---

``gast`` provides the same API as the ``ast`` module. All functions and classes
from the ``ast`` module are also available in the ``gast`` module, and work
accordingly on the ``gast`` tree.

Three notable exceptions:

1. ``gast.parse`` directly produces a GAST node. It's equivalent to running
       ``gast.ast_to_gast`` on the output of ``ast.parse``.

2. ``gast.dump`` dumps the ``gast`` common representation, not the original
       one.

3. ``gast.gast_to_ast`` and ``gast.ast_to_gast`` can be used to convert
       from one ast to the other, back and forth.

Version Compatibility
---------------------

GAST is tested using ``tox`` and Travis on the following Python versions:

- 2.7
- 3.4
- 3.5
- 3.6
- 3.7
- 3.8
- 3.9
- 3.10
- 3.11


AST Changes
-----------


Python3
*******

The AST used by GAST is the same as the one used in Python3.9, with the
notable exception of ``ast.arg`` being replaced by an ``ast.Name`` with an
``ast.Param`` context.

The ``name`` field of ``ExceptHandler`` is represented as an ``ast.Name`` with
an ``ast.Store`` context and not a ``str``.

For minor version before 3.9, please note that ``ExtSlice`` and ``Index`` are
not used.

For minor version before 3.8, please note that ``Ellipsis``, ``Num``, ``Str``,
``Bytes`` and ``NamedConstant`` are represented as ``Constant``.

Python2
*******

To cope with Python3 features, several nodes from the Python2 AST are extended
with some new attributes/children, or represented by different nodes

- ``ModuleDef`` nodes have a ``type_ignores`` attribute.

- ``FunctionDef`` nodes have a ``returns`` attribute and a ``type_comment``
  attribute.

- ``ClassDef`` nodes have a ``keywords`` attribute.

- ``With``'s ``context_expr`` and ``optional_vars`` fields are hold in a
  ``withitem`` object.

- ``For`` nodes have a ``type_comment`` attribute.

- ``Raise``'s ``type``, ``inst`` and ``tback`` fields are hold in a single
  ``exc`` field, using the transformation ``raise E, V, T => raise E(V).with_traceback(T)``.

- ``TryExcept`` and ``TryFinally`` nodes are merged in the ``Try`` node.

- ``arguments`` nodes have a ``kwonlyargs`` and ``kw_defaults`` attributes.

- ``Call`` nodes loose their ``starargs`` attribute, replaced by an
  argument wrapped in a ``Starred`` node. They also loose their ``kwargs``
  attribute, wrapped in a ``keyword`` node with the identifier set to
  ``None``, as done in Python3.

- ``comprehension`` nodes have an ``async`` attribute (that is always set
  to 0).

- ``Ellipsis``, ``Num`` and ``Str`` nodes are represented as ``Constant``.

- ``Subscript`` which don't have any ``Slice`` node as ``slice`` attribute (and
  ``Ellipsis`` are not ``Slice`` nodes) no longer hold an ``ExtSlice`` but an
  ``Index(Tuple(...))`` instead.


Pit Falls
*********

- In Python3, ``None``, ``True`` and ``False`` are parsed as ``Constant``
  while they are parsed as regular ``Name`` in Python2.

ASDL
****

This closely matches the one from https://docs.python.org/3/library/ast.html#abstract-grammar, with a few
trade-offs to cope with legacy ASTs.

.. code::

    -- ASDL's six builtin types are identifier, int, string, bytes, object, singleton

    module Python
    {
        mod = Module(stmt* body, type_ignore *type_ignores)
            | Interactive(stmt* body)
            | Expression(expr body)
            | FunctionType(expr* argtypes, expr returns)

            -- not really an actual node but useful in Jython's typesystem.
            | Suite(stmt* body)

        stmt = FunctionDef(identifier name, arguments args,
                           stmt* body, expr* decorator_list, expr? returns,
                           string? type_comment, type_param* type_params)
              | AsyncFunctionDef(identifier name, arguments args,
                                 stmt* body, expr* decorator_list, expr? returns,
                                 string? type_comment, type_param* type_params)

              | ClassDef(identifier name,
                 expr* bases,
                 keyword* keywords,
                 stmt* body,
                 expr* decorator_list,
                 type_param* type_params)
              | Return(expr? value)

              | Delete(expr* targets)
              | Assign(expr* targets, expr value, string? type_comment)
              | TypeAlias(expr name, type_param* type_params, expr value)
              | AugAssign(expr target, operator op, expr value)
              -- 'simple' indicates that we annotate simple name without parens
              | AnnAssign(expr target, expr annotation, expr? value, int simple)

              -- not sure if bool is allowed, can always use int
              | Print(expr? dest, expr* values, bool nl)

              -- use 'orelse' because else is a keyword in target languages
              | For(expr target, expr iter, stmt* body, stmt* orelse, string? type_comment)
              | AsyncFor(expr target, expr iter, stmt* body, stmt* orelse, string? type_comment)
              | While(expr test, stmt* body, stmt* orelse)
              | If(expr test, stmt* body, stmt* orelse)
              | With(withitem* items, stmt* body, string? type_comment)
              | AsyncWith(withitem* items, stmt* body, string? type_comment)

              | Match(expr subject, match_case* cases)

              | Raise(expr? exc, expr? cause)
              | Try(stmt* body, excepthandler* handlers, stmt* orelse, stmt* finalbody)
              | TryStar(stmt* body, excepthandler* handlers, stmt* orelse, stmt* finalbody)
              | Assert(expr test, expr? msg)

              | Import(alias* names)
              | ImportFrom(identifier? module, alias* names, int? level)

              -- Doesn't capture requirement that locals must be
              -- defined if globals is
              -- still supports use as a function!
              | Exec(expr body, expr? globals, expr? locals)

              | Global(identifier* names)
              | Nonlocal(identifier* names)
              | Expr(expr value)
              | Pass | Break | Continue

              -- XXX Jython will be different
              -- col_offset is the byte offset in the utf8 string the parser uses
              attributes (int lineno, int col_offset)

              -- BoolOp() can use left & right?
        expr = BoolOp(boolop op, expr* values)
             | NamedExpr(expr target, expr value)
             | BinOp(expr left, operator op, expr right)
             | UnaryOp(unaryop op, expr operand)
             | Lambda(arguments args, expr body)
             | IfExp(expr test, expr body, expr orelse)
             | Dict(expr* keys, expr* values)
             | Set(expr* elts)
             | ListComp(expr elt, comprehension* generators)
             | SetComp(expr elt, comprehension* generators)
             | DictComp(expr key, expr value, comprehension* generators)
             | GeneratorExp(expr elt, comprehension* generators)
             -- the grammar constrains where yield expressions can occur
             | Await(expr value)
             | Yield(expr? value)
             | YieldFrom(expr value)
             -- need sequences for compare to distinguish between
             -- x < 4 < 3 and (x < 4) < 3
             | Compare(expr left, cmpop* ops, expr* comparators)
             | Call(expr func, expr* args, keyword* keywords)
             | Repr(expr value)
             | FormattedValue(expr value, int? conversion, expr? format_spec)
             | JoinedStr(expr* values)
             | Constant(constant value, string? kind)

             -- the following expression can appear in assignment context
             | Attribute(expr value, identifier attr, expr_context ctx)
             | Subscript(expr value, slice slice, expr_context ctx)
             | Starred(expr value, expr_context ctx)
             | Name(identifier id, expr_context ctx, expr? annotation,
                    string? type_comment)
             | List(expr* elts, expr_context ctx)
             | Tuple(expr* elts, expr_context ctx)

              -- col_offset is the byte offset in the utf8 string the parser uses
              attributes (int lineno, int col_offset)

        expr_context = Load | Store | Del | AugLoad | AugStore | Param

        slice = Slice(expr? lower, expr? upper, expr? step)
              | ExtSlice(slice* dims)
              | Index(expr value)

        boolop = And | Or

        operator = Add | Sub | Mult | MatMult | Div | Mod | Pow | LShift
                     | RShift | BitOr | BitXor | BitAnd | FloorDiv

        unaryop = Invert | Not | UAdd | USub

        cmpop = Eq | NotEq | Lt | LtE | Gt | GtE | Is | IsNot | In | NotIn

        comprehension = (expr target, expr iter, expr* ifs, int is_async)

        excepthandler = ExceptHandler(expr? type, expr? name, stmt* body)
                        attributes (int lineno, int col_offset, int? end_lineno, int? end_col_offset)

        arguments = (expr* args, expr* posonlyargs, expr? vararg, expr* kwonlyargs,
                     expr* kw_defaults, expr? kwarg, expr* defaults)

        -- keyword arguments supplied to call (NULL identifier for **kwargs)
        keyword = (identifier? arg, expr value)

        -- import name with optional 'as' alias.
        alias = (identifier name, identifier? asname)
                attributes (int lineno, int col_offset, int? end_lineno, int? end_col_offset)

        withitem = (expr context_expr, expr? optional_vars)

        match_case = (pattern pattern, expr? guard, stmt* body)

        pattern = MatchValue(expr value)
                | MatchSingleton(constant value)
                | MatchSequence(pattern* patterns)
                | MatchMapping(expr* keys, pattern* patterns, identifier? rest)
                | MatchClass(expr cls, pattern* patterns, identifier* kwd_attrs, pattern* kwd_patterns)

                | MatchStar(identifier? name)
                -- The optional "rest" MatchMapping parameter handles capturing extra mapping keys

                | MatchAs(pattern? pattern, identifier? name)
                | MatchOr(pattern* patterns)

                 attributes (int lineno, int col_offset, int end_lineno, int end_col_offset)

        type_ignore = TypeIgnore(int lineno, string tag)

         type_param = TypeVar(identifier name, expr? bound)
                    | ParamSpec(identifier name)
                    | TypeVarTuple(identifier name)
                    attributes (int lineno, int col_offset, int end_lineno, int end_col_offset)
    }


Reporting Bugs
--------------

Bugs can be reported through `GitHub issues <https://github.com/serge-sans-paille/gast/issues>`_.

Reporting Security Issues
-------------------------

If for some reason, you think your bug is security-related and should be subject
to responsible disclosure, don't hesitate to `contact the maintainer
<mailto:serge.guelton@telecom-bretagne.eu>`_ directly.
