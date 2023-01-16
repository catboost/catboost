### lowercasing

#### Description

Convert tokens to lower case.

**Data types**

{{ python-type--bool }}

**Default value**

Tokens are not converted to lower case

### lemmatizing

#### Description

Perform lemmatization on tokens.

**Data types**

{{ python-type--bool }}

**Default value**

Lemmatization is not performed

### number_process_policy

#### Description

The strategy to process numeric tokens. Possible values:

- {{ values__numberprocesspolicy__Skip }} — Skip all numeric tokens.
- {{ values__numberprocesspolicy__LeaveAsIs }} — Leave all numeric tokens as is.
- {{ values__numberprocesspolicy__Replace }} — Replace all numeric tokens with a single special token. This token in specified in the `number_token` parameter.

**Data types**

{{ python-type--string }}

**Default value**

{{ values__numberprocesspolicy__LeaveAsIs }}

### number_token

#### Description

The special token that is used to replace all numeric tokens with.

This option can be used if the selected numeric tokens processing strategy is {{ values__numberprocesspolicy__Replace }}.

This token is not converted to lower case regardless of the value of the set value of the `lowercasing` parameter.

**Data types**

{{ python-type--string }}

**Default value**

Numeric tokens are left as is

### separator_type

#### Description

The tokenization method. Possible values:

- `ByDelimiter` — Split by delimiter.
- `BySense` — Try to split the string by sense.

**Data types**

{{ python-type--string }}

**Default value**

`ByDelimiter` with the delimiter set to “ ” (whitespace)

### delimiter

#### Description

The symbol that is considered to be the delimiter.

Should be used if the `separator_type` parameter is set to “ ”.

**Data types**

{{ python-type--string }}

**Default value**

“ ” (whitespace)

### split_by_set

#### Description

Use each single character in the `delimiter` option as an individual delimiter.

Use this parameter to apply multiple delimiters.

**Data types**

{{ python-type--bool }}

**Default value**

False (the whole string specified in the `delimiter` parameter is considered as the delimiter)

### skip_empty

#### Description

Skip all empty tokens.

**Data types**

{{ python-type--bool }}

**Default value**

True (empty tokens are skipped)

### token_types

#### Description

The types of tokens that should be kept after the tokenization.

Should be used if the `separator_type` parameter is set to `BySense`.

Possible values:
- `Word`
- `Number`
- `Punctuation`
- `SentenceBreak`
- `ParagraphBreak`
- `Unknown`

**Data types**

{{ python-type--list }}

**Default value**

All supported types of tokens are kept

### sub_tokens_policy

#### Description

The subtokens processing policy.

Should be used if the `separator_type` parameter is set to `BySense`.

Possible values:
- `SingleToken` — All subtokens are interpreted as a single token.
- `SeveralTokens` — All subtokens are interpreted as several tokens.

**Data types**

{{ python-type--string }}

**Default value**

`SingleToken`

### languages

#### Description

The list of languages to use.

Should be used if the `separator_type` parameter is set to `BySense`.

**Data types**

{{ python-type--list-of-strings }}

**Default value**

All available languages are used (significantly slows down the procedure)
