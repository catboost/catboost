MarkupSafe
==========

MarkupSafe implements a text object that escapes characters so it is
safe to use in HTML and XML. Characters that have special meanings are
replaced so that they display as the actual characters. This mitigates
injection attacks, meaning untrusted user input can safely be displayed
on a page.


Installing
----------

Install and update using `pip`_:

.. code-block:: text

    pip install -U MarkupSafe

.. _pip: https://pip.pypa.io/en/stable/getting-started/


Examples
--------

.. code-block:: pycon

    >>> from markupsafe import Markup, escape

    >>> # escape replaces special characters and wraps in Markup
    >>> escape("<script>alert(document.cookie);</script>")
    Markup('&lt;script&gt;alert(document.cookie);&lt;/script&gt;')

    >>> # wrap in Markup to mark text "safe" and prevent escaping
    >>> Markup("<strong>Hello</strong>")
    Markup('<strong>hello</strong>')

    >>> escape(Markup("<strong>Hello</strong>"))
    Markup('<strong>hello</strong>')

    >>> # Markup is a str subclass
    >>> # methods and operators escape their arguments
    >>> template = Markup("Hello <em>{name}</em>")
    >>> template.format(name='"World"')
    Markup('Hello <em>&#34;World&#34;</em>')


Donate
------

The Pallets organization develops and supports MarkupSafe and other
popular packages. In order to grow the community of contributors and
users, and allow the maintainers to devote more time to the projects,
`please donate today`_.

.. _please donate today: https://palletsprojects.com/donate


Links
-----

-   Documentation: https://markupsafe.palletsprojects.com/
-   Changes: https://markupsafe.palletsprojects.com/changes/
-   PyPI Releases: https://pypi.org/project/MarkupSafe/
-   Source Code: https://github.com/pallets/markupsafe/
-   Issue Tracker: https://github.com/pallets/markupsafe/issues/
-   Chat: https://discord.gg/pallets
