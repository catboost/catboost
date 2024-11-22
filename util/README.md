# Coding style

Style guide for the util folder is a stricter version of
[general style guide](https://docs.yandex-team.ru/arcadia-cpp/cpp_style_guide)
(mostly in terms of ambiguity resolution).

 * all {} must be in K&R style
 * &, * tied closer to a type, not to variable
 * always use `using` not `typedef`
 * _ at the end of private data member of a class - `First_`, `Second_`
 * every .h file must be accompanied with corresponding .cpp to avoid a leakage and check that it is self contained
 * prohibited to use `printf`-like functions


Things declared in the general style guide, which sometimes are missed:

 * `template <`, not `template<`
 * `noexcept`, not `throw ()` nor `throw()`, not required for destructors
 * indents inside `namespace` same as inside `class`


Requirements for a new code (and for corrections in an old code which involves change of behaviour) in util:

 * presence of UNIT-tests
 * presence of comments in Doxygen style
 * accessors without Get prefix (`Length()`, but not `GetLength()`)

This guide is not a mandatory as there is the general style guide.
Nevertheless if it is not followed, then a next `ya style .` run in the util folder will undeservedly update authors of some lines of code.

Thus before a commit it is recommended to run `ya style .` in the util folder.


Don't forget to run tests from folder `tests`: `ya make -t tests`

**Note:** tests are designed to run using `autocheck/` solution.

# Submitting a patch

In order to make a commit, you have to get approval from one of
[util](https://arcanum.yandex-team.ru/arc/trunk/arcadia/groups/util) members.

If no comments have been received withing 1–2 days, it is OK
to send a graceful ping into [Igni et ferro](https://wiki.yandex-team.ru/ignietferro/) chat.

Certain exceptions apply. The following trivial changes do not need to be reviewed:

* docs, comments, typo fixes,
* renaming of an internal variable to match the styleguide.

Whenever a breaking change happens to accidentally land into trunk, reverting it does not need to be reviewed.

## Stale/abandoned review request policy

Sometimes review requests are neither merged nor discarded, and stay in review request queue forever.
To limit the incoming review request queue size, util reviewers follow these rules:

- A review request is considered stale if it is not updated by its author for at least 3 months, or if its author has left Yandex.
- A stale review request may be marked as discarded by util reviewers.

Review requests discarded as stale may be reopened or resubmitted by any committer willing to push them to completion.

**Note:** It's an author's duty to push the review request to completion.
If util reviewers stop responding to updates, they should be politely pinged via appropriate means of communication.
