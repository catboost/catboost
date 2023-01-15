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

.. _pip: https://pip.pypa.io/en/stable/quickstart/


Examples
--------

.. code-block:: pycon

    >>> from markupsafe import Markup, escape
    >>> # escape replaces special characters and wraps in Markup
    >>> escape('<script>alert(document.cookie);</script>')
    Markup(u'&lt;script&gt;alert(document.cookie);&lt;/script&gt;')
    >>> # wrap in Markup to mark text "safe" and prevent escaping
    >>> Markup('<strong>Hello</strong>')
    Markup('<strong>hello</strong>')
    >>> escape(Markup('<strong>Hello</strong>'))
    Markup('<strong>hello</strong>')
    >>> # Markup is a text subclass (str on Python 3, unicode on Python 2)
    >>> # methods and operators escape their arguments
    >>> template = Markup("Hello <em>%s</em>")
    >>> template % '"World"'
    Markup('Hello <em>&#34;World&#34;</em>')


Donate
------

The Pallets organization develops and supports MarkupSafe and other
libraries that use it. In order to grow the community of contributors
and users, and allow the maintainers to devote more time to the
projects, `please donate today`_.

.. _please donate today: https://palletsprojects.com/donate


Links
-----

*   Website: https://palletsprojects.com/p/markupsafe/
*   Documentation: https://markupsafe.palletsprojects.com/
*   License: `BSD-3-Clause <https://github.com/pallets/markupsafe/blob/master/LICENSE.rst>`_
*   Releases: https://pypi.org/project/MarkupSafe/
*   Code: https://github.com/pallets/markupsafe
*   Issue tracker: https://github.com/pallets/markupsafe/issues
*   Test status:

    *   Linux, Mac: https://travis-ci.org/pallets/markupsafe
    *   Windows: https://ci.appveyor.com/project/pallets/markupsafe

*   Test coverage: https://codecov.io/gh/pallets/markupsafe
