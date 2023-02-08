==========
Hypothesis
==========

Hypothesis is an advanced testing library for Python. It lets you write tests which
are parametrized by a source of examples, and then generates simple and comprehensible
examples that make your tests fail. This lets you find more bugs in your code with less
work.

e.g.

.. code-block:: python

  @given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1))
  def test_mean(xs):
      assert min(xs) <= mean(xs) <= max(xs)

.. code-block::

  Falsifying example: test_mean(
    xs=[1.7976321109618856e+308, 6.102390043022755e+303]
  )

Hypothesis is extremely practical and advances the state of the art of
unit testing by some way. It's easy to use, stable, and powerful. If
you're not using Hypothesis to test your project then you're missing out.

------------------------
Quick Start/Installation
------------------------

If you just want to get started:

.. code-block::

  pip install hypothesis


-----------------
Links of interest
-----------------

The main Hypothesis site is at `hypothesis.works <https://hypothesis.works/>`_, and contains a lot
of good introductory and explanatory material.

Extensive documentation and examples of usage are `available at readthedocs <https://hypothesis.readthedocs.io/en/latest/>`_.

If you want to talk to people about using Hypothesis, `we have both an IRC channel
and a mailing list <https://hypothesis.readthedocs.io/en/latest/community.html>`_.

If you want to receive occasional updates about Hypothesis, including useful tips and tricks, there's a
`TinyLetter mailing list to sign up for them <https://tinyletter.com/DRMacIver/>`_.

If you want to contribute to Hypothesis, `instructions are here <https://github.com/HypothesisWorks/hypothesis-python/blob/master/CONTRIBUTING.rst>`_.

If you want to hear from people who are already using Hypothesis, some of them `have written
about it <https://hypothesis.readthedocs.io/en/latest/endorsements.html>`_.

If you want to create a downstream package of Hypothesis, please read `these guidelines for packagers <https://hypothesis.readthedocs.io/en/latest/packaging.html>`_.
