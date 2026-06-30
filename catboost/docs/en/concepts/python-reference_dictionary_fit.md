# fit

{% include [descriptions-fit](../_includes/work_src/reusage-tokenizer/fit.md) %}


## {{ dl--invoke-format }} {#call-format}

```
fit(data,
    tokenizer=None,
    verbose=False)
```

## {{ dl--parameters }} {#parameters}

### data

#### Description
The description is different for each group of possible types.

**Data types**

{% cut "{{ python-type--string }}" %}

The path to the input file with text data.

The text must be input from a file if the selected dictionary type is {{ dictionary__dictionary-type__Bpe }}.

{% endcut %}


{% cut "{{ python-type--list }}, {{ python-type--numpy-ndarray }}, {{ python-type--pandasSeries }}" %}

One- or two-dimensional array-like input text data.

{% endcut %}


**Default value**

{{ loss-functions__params__q__default }}

### tokenizer

#### Description

The tokenizer for text processing.

If this parameter is specified and a one-dimensional data is input, each element in this list is considered a sentence and is tokenized.

**Data types**

[Tokenizer](../concepts/python-reference_tokenizer.md)

Default value: None (the input data is considered tokenized)

### verbose

#### Description

Output the progress information.

**Data types**

{{ python-type--bool }}

**Default value**

False

## {{ dl--output-format }} {#return-value}

_catboost.Dictionary

## {{ input_data__title__example }} {#example}

```
from catboost.text_processing import Dictionary

dictionary = Dictionary(occurence_lower_bound=0,
                        )\
    .fit(["but", "as", "the", "riper", "should", "by", "time", "decease",\
          "his", "tender", "heir", "might", "bear", "his", "memory"])

tokens = dictionary.get_top_tokens(14)
print tokens
dictionary.save("frequency_dict_path")
```

In this example, the [get_top_tokens](python-reference_dictionary_get_top_tokens.md) function is used to output the list of tokens.

Output:

```no-highlight
['his', 'as', 'bear', 'but', 'by', 'decease', 'heir', 'memory', 'might', 'riper', 'should', 'tender', 'the', 'time']
```

## Reading data from file example {#example-from-file}

#### Contents of the `dictionary-wtih-tokenizer-text` input file

```no-highlight
From fairest creatures we desire increase,
That thereby beauty's rose might never die,
But as the riper should by time decease,
His tender heir might bear his memory:
But thou contracted to thine own bright eyes,
Feed'st thy light's flame with self-substantial fuel,
Making a famine where abundance lies,
Thy self thy foe, to thy sweet self too cruel:
Thou that art now the world's fresh ornament,
And only herald to the gaudy spring,
Within thine own bud buriest thy content,
And, tender churl, mak'st waste in niggarding:
Pity the world, or else this glutton be,
To eat the world's due, by the grave and thee.
```

The following code is used to train the dictionary:

```python
from catboost.text_processing import Dictionary, Tokenizer

tokenized = Tokenizer()

dictionary = Dictionary(occurence_lower_bound=0)\
    .fit("dictionary-wtih-tokenizer-text",\
    tokenizer=tokenized)

dictionary.save("frequency_dict_path")

```

In this example, a constructor of the [Tokenizer](python-reference_tokenizer.md) class is used.W

Output:
```bash
{"end_of_word_token_policy":"Insert","skip_step":"0","start_token_id":"0","token_level_type":"Word","dictionary_format":"id_count_token","end_of_sentence_token_policy":"Skip","gram_order":"1"}
88
0	6	the
1	4	thy
2	3	to
3	2	But
4	2	by
5	2	might
6	2	own
7	2	self
8	2	tender
9	2	thine
10	2	world's
11	1	And
12	1	And,
13	1	Feed'st
14	1	From
15	1	His
16	1	Making
17	1	Pity
18	1	That
19	1	Thou
20	1	Thy
21	1	To
22	1	Within
23	1	a
24	1	abundance
25	1	and
26	1	art
27	1	as
28	1	be,
29	1	bear
30	1	beauty's
31	1	bright
32	1	bud
33	1	buriest
34	1	churl,
35	1	content,
36	1	contracted
37	1	creatures
38	1	cruel:
39	1	decease,
40	1	desire
41	1	die,
42	1	due,
43	1	eat
44	1	else
45	1	eyes,
46	1	fairest
47	1	famine
48	1	flame
49	1	foe,
50	1	fresh
51	1	fuel,
52	1	gaudy
53	1	glutton
54	1	grave
55	1	heir
56	1	herald
57	1	his
58	1	in
59	1	increase,
60	1	lies,
61	1	light's
62	1	mak'st
63	1	memory:
64	1	never
65	1	niggarding:
66	1	now
67	1	only
68	1	or
69	1	ornament,
70	1	riper
71	1	rose
72	1	self-substantial
73	1	should
74	1	spring,
75	1	sweet
76	1	that
77	1	thee.
78	1	thereby
79	1	this
80	1	thou
81	1	time
82	1	too
83	1	waste
84	1	we
85	1	where
86	1	with
87	1	world,
```
