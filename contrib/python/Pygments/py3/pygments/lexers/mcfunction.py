"""
    pygments.lexers.mcfunction
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

    Lexers for MCFunction and related languages.

    :copyright: Copyright 2006-2022 by the Pygments team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

from pygments.lexer import RegexLexer, default, include, bygroups
from pygments.token import (Comment, Keyword, Literal, Name, Number,
                            Operator, Punctuation, String, Text, Token,
                            Whitespace)


__all__ = ['SNBTLexer', 'MCFunctionLexer']


class SNBTLexer(RegexLexer):
    """Lexer for stringified NBT, a data format used in Minecraft

    .. versionadded:: 2.12.0
    """

    name = "SNBT"
    url = "https://minecraft.fandom.com/wiki/NBT_format"
    aliases = ["snbt"]
    filenames = ["*.snbt"]
    mimetypes = ["text/snbt"]

    tokens = {
        "root": [
            # We only look for the open bracket here since square bracket
            #  is only valid in NBT pathing (which is a mcfunction idea).
            (r"\{", Punctuation, "compound"),
            (r"[^\{]+", Text),
        ],

        "whitespace": [
            (r"\s+", Whitespace),
        ],

        "operators": [
            (r"[,:;]", Punctuation),
        ],

        "literals": [
            (r"(true|false)", Keyword.Constant),
            (r"-?\d+[eE]-?\d+", Number.Float),
            (r"-?\d*\.\d+[fFdD]?", Number.Float),
            (r"-?\d+[bBsSlLfFdD]?", Number.Integer),

            # Separate states for both types of strings so they don't entangle
            (r'"', String.Double, "literals.string_double"),
            (r"'", String.Single, "literals.string_single"),
        ],
        "literals.string_double": [
            (r"\\.", String.Escape),
            (r'[^\\"\n]+', String.Double),
            (r'"', String.Double, "#pop"),
        ],
        "literals.string_single": [
            (r"\\.", String.Escape),
            (r"[^\\'\n]+", String.Single),
            (r"'", String.Single, "#pop"),
        ],

        "compound": [
            # this handles the unquoted snbt keys
            #  note: stringified keys still work
            (r"[A-Z_a-z]+", Name.Attribute),
            include("operators"),
            include("whitespace"),
            include("literals"),
            (r"\{", Punctuation, "#push"),
            (r"\[", Punctuation, "list"),
            (r"\}", Punctuation, "#pop"),
        ],

        "list": [
            (r"[A-Z_a-z]+", Name.Attribute),
            include("literals"),
            include("operators"),
            include("whitespace"),
            (r"\[", Punctuation, "#push"),
            (r"\{", Punctuation, "compound"),
            (r"\]", Punctuation, "#pop"),
        ],
    }


class MCFunctionLexer(RegexLexer):
    """Lexer for the mcfunction scripting language used in Minecraft
    Modelled somewhat after the `GitHub mcfunction grammar <https://github.com/Arcensoth/language-mcfunction>`_.

    .. versionadded:: 2.12.0
    """

    name = "MCFunction"
    url = "https://minecraft.fandom.com/wiki/Commands"
    aliases = ["mcfunction", "mcf"]
    filenames = ["*.mcfunction"]
    mimetypes = ["text/mcfunction"]

    # Used to denotate the start of a block comment, borrowed from Github's mcfunction
    _block_comment_prefix = "[>!]"

    tokens = {
        "root": [
            include("names"),
            include("comments"),
            include("literals"),
            include("whitespace"),
            include("property"),
            include("operators"),
            include("selectors"),
        ],

        "names": [
            # The start of a command (either beginning of line OR after the run keyword)
            #  We don't encode a list of keywords since mods, plugins, or even pre-processors
            #  may add new commands, so we have a 'close-enough' regex which catches them.
            (r"^(\s*)([a-z_]+)", bygroups(Whitespace, Name.Builtin)),
            (r"(?<=run)\s+[a-z_]+", Name.Builtin),

            # UUID
            (
                r"\b[0-9a-fA-F]+(?:-[0-9a-fA-F]+){4}\b",
                Name.Variable,
            ),
            include("resource-name"),
            # normal command names and scoreboards
            #  there's no way to know the differences unfortuntely
            (r"[A-Za-z_][A-Za-z0-9_.#%$]+", Keyword.Constant),
            (r"[#%$][A-Za-z0-9_.#%$]+", Name.Variable.Magic),
        ],

        "resource-name": [
            (
                # resource names have to be lowercase
                r"#?[a-z_][a-z_.-]*:[a-z0-9_./-]+",
                Name.Function,
            ),
            (
                # similar to above except optional `:``
                #  a `/` must be present "somewhere"
                r"#?[a-z0-9_\.\-]+\/[a-z0-9_\.\-\/]+",
                Name.Function,
            )
        ],

        "whitespace": [
            (r"\s+", Whitespace),
        ],

        "comments": [
            (
                rf"^\s*(#{_block_comment_prefix})",
                Comment.Multiline,
                (
                    "comments.block",
                    "comments.block.emphasized",
                ),
            ),
            (r"#.*$", Comment.Single),
        ],
        "comments.block": [
            (rf"^\s*#{_block_comment_prefix}", Comment.Multiline, "comments.block.emphasized"),
            (r"^\s*#", Comment.Multiline, "comments.block.normal"),
            default("#pop"),
        ],
        "comments.block.normal": [
            include("comments.block.special"),
            (r"\S+", Comment.Multiline),
            (r"\n", Text, "#pop"),
            include("whitespace"),
        ],
        "comments.block.emphasized": [
            include("comments.block.special"),
            (r"\S+", String.Doc),
            (r"\n", Text, "#pop"),
            include("whitespace"),
        ],
        "comments.block.special": [
            # Params
            (r"@\S+", Name.Decorator),

            include("resource-name"),

            # Scoreboard player names
            (r"[#%$][A-Za-z0-9_.#%$]+", Name.Variable.Magic),
        ],

        "operators": [
            (r"[\-~%^?!+*<>\\/|&=.]", Operator),
        ],

        "literals": [
            (r"\.\.", Literal),
            (r"(true|false)", Keyword.Pseudo),

            # these are like unquoted strings and appear in many places
            (r"[A-Za-z_]+", Name.Variable.Class),

            (r"[0-7]b", Number.Byte),
            (r"[+-]?\d*\.?\d+([eE]?[+-]?\d+)?[df]?\b", Number.Float),
            (r"[+-]?\d+\b", Number.Integer),
            (r'"', String.Double, "literals.string-double"),
            (r"'", String.Single, "literals.string-single"),
        ],
        "literals.string-double": [
            (r"\\.", String.Escape),
            (r'[^\\"\n]+', String.Double),
            (r'"', String.Double, "#pop"),
        ],
        "literals.string-single": [
            (r"\\.", String.Escape),
            (r"[^\\'\n]+", String.Single),
            (r"'", String.Single, "#pop"),
        ],

        "selectors": [
            (r"@[a-z]", Name.Variable),
        ],
 

        ## Generic Property Container
        # There are several, differing instances where the language accepts
        #  specific contained keys or contained key, value pairings.
        # 
        # Property Maps:
        # - Starts with either `[` or `{`
        # - Key separated by `:` or `=`
        # - Deliminated by `,`
        # 
        # Property Lists:
        # - Starts with `[`
        # - Deliminated by `,`
        # 
        # For simplicity, these patterns match a generic, nestable structure
        #  which follow a key, value pattern. For normal lists, there's only keys.
        # This allow some "illegal" structures, but we'll accept those for
        #  sake of simplicity
        # 
        # Examples:
        # - `[facing=up, powered=true]` (blockstate)
        # - `[name="hello world", nbt={key: 1b}]` (selector + nbt)
        # - `[{"text": "value"}, "literal"]` (json)
        ##
        "property": [
            # This state gets included in root and also several substates
            # We do this to shortcut the starting of new properties
            #  within other properties. Lists can have sublists and compounds
            #  and values can start a new property (see the `difficult_1.txt`
            #  snippet).
            (r"\{", Punctuation, ("property.curly", "property.key")),
            (r"\[", Punctuation, ("property.square", "property.key")),
        ],
        "property.curly": [
            include("whitespace"),
            include("property"),
            (r"\}", Punctuation, "#pop"),
        ],
        "property.square": [
            include("whitespace"),
            include("property"),
            (r"\]", Punctuation, "#pop"),

            # lists can have sequences of items
            (r",", Punctuation),
        ],
        "property.key": [
            include("whitespace"),

            # resource names (for advancements)
            #  can omit `:` to default `minecraft:`
            # must check if there is a future equals sign if `:` is in the name
            (r"#?[a-z_][a-z_\.\-]*\:[a-z0-9_\.\-/]+(?=\s*\=)", Name.Attribute, "property.delimiter"),
            (r"#?[a-z_][a-z0-9_\.\-/]+", Name.Attribute, "property.delimiter"),

            # unquoted NBT key
            (r"[A-Za-z_\-\+]+", Name.Attribute, "property.delimiter"),

            # quoted JSON or NBT key
            (r'"', Name.Attribute, "property.delimiter", "literals.string-double"),
            (r"'", Name.Attribute, "property.delimiter", "literals.string-single"),

            # index for a list
            (r"-?\d+", Number.Integer, "property.delimiter"),

            default("#pop"),
        ],
        "property.key.string-double": [
            (r"\\.", String.Escape),
            (r'[^\\"\n]+', Name.Attribute),
            (r'"', Name.Attribute, "#pop"),
        ],
        "property.key.string-single": [
            (r"\\.", String.Escape),
            (r"[^\\'\n]+", Name.Attribute),
            (r"'", Name.Attribute, "#pop"),
        ],
        "property.delimiter": [
            include("whitespace"),
            
            (r"[:=]!?", Punctuation, "property.value"),
            (r",", Punctuation),

            default("#pop"),
        ],
        "property.value": [
            include("whitespace"),

            # unquoted resource names are valid literals here
            (r"#?[a-z_][a-z_\.\-]*\:[a-z0-9_\.\-/]+", Name.Tag),
            (r"#?[a-z_][a-z0-9_\.\-/]+", Name.Tag),

            include("literals"),
            include("property"),

            default("#pop"),
        ],
    }
