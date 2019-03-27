# Dictionary
## Info About
There are several types of dictionaries, with which you can do the following things
- Build a dictionary for the text corpus. Dictionary can be build on the basis of the words, word n-grams and letter n-grams (further tokens).
- Apply a dictionary to the text corpus (Replace tokens with the IDs).
- Get the top of most frequent tokens.

**Dictionary Types:**
- **Frequency Based Dictionary**: This dictionary is built taking into account only the most frequent tokens (The dictionary size is set via parameter).
- **BPE Dictionary**: This dictionary is build taking into account the most frequent tokens and then combine the most frequent tokens pairs into single new token. For more information. Refer to https://arxiv.org/pdf/1508.07909.pdf for more details.

## Structure
- `dictionary.h` - is common interface for all dictionaries.
- `dictionary_builder.h` - is class for Frequency Based Dictionary building
- `bpe_builder.h` - is class for BPE Dictionary building.

## Dictionary Methods
 - `TTokenId Apply(TStringBuf token)` - Apply dictionary to single token.
 - `void Apply(TConstArrayRef<TString> tokens, TVector<TTokenId>* tokensIds)` - Apply dictionary to several tokens.
 - `ui32 Size()` - Get dictionary Size.
 - `TString GetToken(TTokenId tokenId)` - Get token by id.
 - `ui64 GetCount(TTokenId tokenId)` - Get token count by id
 - `TVector<TString> GetTopTokens(ui32 topSize = 10)` - Get the top of most frequent tokens.
 - `void ClearStatsData()` - Clear internal class fields for less memory consumption (In this case, `GetToken`, `GetCount`, `GetTopTokens` methods will not work.)
 - `TTokenId GetUnknownTokenId()` - Get unknown token id.
 - `TTokenId GetEndOfSentenceTokenId()` - Get end of sentence token id.
 - `TTokenId GetMinUnusedTokenId()` - Get min unused token id.
 - `void Save(IOutputStream* stream)` - Save dictionary to the stream.
 - `static THolder<IDictionary> Load(IInputStream* stream)` - Load dictionary from stream.

## Dictionary Builder Methods
 - `void Add(TStringBuf token, ui64 weight = 1)` - Add single token to dictionary.
 - `void Add(TConstArrayRef<TString> tokens, ui64 weight = 1);` - Add several tokens to dictionary.
 - `THolder<TDictionary> FinishBuilding()` - Finish building and get pointer to dictionary.

## Frequency Based Dictionary Builder Options
- `OccurrenceLowerBound` [`ui64`] - The lower bound of token occurrences in the text to include it in the dictionary.
- `MaxDictionarySize` [`int`] - The max dictionary size.

## Frequency Based Dictionary Options
- `TokenLevelType` [`ETokenLevelType`] - Token level type. With this parameter, you can configure what to choose as a separate token: a word or a letter. Posible values: `Word`, `Letter`.
- `GramOrder` [`ui32`] - Gram Order (1 for Unigram, 2 for Bigram, ...).
- `SkipStep` [`ui32`] - Skip step (1 for 1-skip-gram, ...).
- `StartTokenId` [`ui32`] - Initial shift for tokenId.

## BPE Dictionary Options
- `NumUnits` [`ui32`] - Number of token pairs that combine into single new token.
- `SkipUnknown` [`bool`] - Skip unknown tokens in building.
