Contributing
------------

libtiff uses a ``.clang-format`` file to enforce code formatting rules.

Automatic code reformatting can be done with pre-commit.

Install (once) pre-commit with ``python -m pip install pre-commit``.

Install it (once) in the libtiff git repository with ``pre-commit install``.

Then the rules defined in the ``.pre-commit-config.yaml`` file will be
enforced at ``git commit`` time, with automatic reformatting.

Due to whole-tree code reformatting done during libtiff 4.5 development,
``git blame`` information might be misleading. To avoid that, you need
to modify your git configuration as following to ignore the revision of
the whole-tree reformatting:
``git config blame.ignoreRevsFile .git-blame-ignore-revs``.
