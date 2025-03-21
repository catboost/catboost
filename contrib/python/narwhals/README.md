# Narwhals

<h1 align="center">
	<img
		width="400"
		alt="narwhals_small"
		src="https://github.com/narwhals-dev/narwhals/assets/33491632/26be901e-5383-49f2-9fbd-5c97b7696f27">
</h1>

[![PyPI version](https://badge.fury.io/py/narwhals.svg)](https://badge.fury.io/py/narwhals)
[![Downloads](https://static.pepy.tech/badge/narwhals/month)](https://pepy.tech/project/narwhals)
[![Trusted publishing](https://img.shields.io/badge/Trusted_publishing-Provides_attestations-bright_green)](https://peps.python.org/pep-0740/)
[![PYPI - Types](https://img.shields.io/pypi/types/narwhals)](https://pypi.org/project/narwhals)

Extremely lightweight and extensible compatibility layer between dataframe libraries!

- **Full API support**: cuDF, Modin, pandas, Polars, PyArrow
- **Lazy-only support**: Dask. Work in progress: DuckDB, Ibis, PySpark.

Seamlessly support all, without depending on any!

- ✅ **Just use** [a subset of **the Polars API**](https://narwhals-dev.github.io/narwhals/api-reference/), no need to learn anything new
- ✅ **Zero dependencies**, Narwhals only uses what
  the user passes in so your library can stay lightweight
- ✅ Separate **lazy** and eager APIs, use **expressions**
- ✅ Support pandas' complicated type system and index, without
  either getting in the way
- ✅ **100% branch coverage**, tested against pandas and Polars nightly builds
- ✅ **Negligible overhead**, see [overhead](https://narwhals-dev.github.io/narwhals/overhead/)
- ✅ Let your IDE help you thanks to **full static typing**, see [typing](https://narwhals-dev.github.io/narwhals/api-reference/typing/)
- ✅ **Perfect backwards compatibility policy**,
  see [stable api](https://narwhals-dev.github.io/narwhals/backcompat/) for how to opt-in

Get started!

- [Read the documentation](https://narwhals-dev.github.io/narwhals/)
- [Chat with us on Discord!](https://discord.gg/V3PqtB4VA4)
- [Join our community call](https://calendar.google.com/calendar/embed?src=27ff6dc5f598c1d94c1f6e627a1aaae680e2fac88f848bda1f2c7946ae74d5ab%40group.calendar.google.com)
- [Read the contributing guide](https://github.com/narwhals-dev/narwhals/blob/main/CONTRIBUTING.md)

<details>
<summary>Table of contents</summary>

- [Narwhals](#narwhals)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Example](#example)
  - [Scope](#scope)
  - [Roadmap](#roadmap)
  - [Used by](#used-by)
  - [Sponsors and institutional partners](#sponsors-and-institutional-partners)
  - [Appears on](#appears-on)
  - [Why "Narwhals"?](#why-narwhals)

</details>

## Installation

- pip (recommended, as it's the most up-to-date)
  ```
  pip install narwhals
  ```
- conda-forge (also fine, but the latest version may take longer to appear)
  ```
  conda install -c conda-forge narwhals
  ```

## Usage

There are three steps to writing dataframe-agnostic code using Narwhals:

1. use `narwhals.from_native` to wrap a pandas/Polars/Modin/cuDF/PyArrow
   DataFrame/LazyFrame in a Narwhals class
2. use the [subset of the Polars API supported by Narwhals](https://narwhals-dev.github.io/narwhals/api-reference/)
3. use `narwhals.to_native` to return an object to the user in its original
   dataframe flavour. For example:

   - if you started with pandas, you'll get pandas back
   - if you started with Polars, you'll get Polars back
   - if you started with Modin, you'll get Modin back (and compute will be distributed)
   - if you started with cuDF, you'll get cuDF back (and compute will happen on GPU)
   - if you started with PyArrow, you'll get PyArrow back

<h1 align="left">
	<img
		width="600"
		alt="narwhals_gif"
		src="https://github.com/user-attachments/assets/88292d3c-6359-4155-973d-d0f8e3fbf5ac">

</h1>

## Example

See the [tutorial](https://narwhals-dev.github.io/narwhals/basics/dataframe/) for several examples!

## Scope

- Do you maintain a dataframe-consuming library?
- Do you have a specific Polars function in mind that you would like Narwhals to have in order to make your work easier?

If you said yes to both, we'd love to hear from you!

## Roadmap

See [roadmap discussion on GitHub](https://github.com/narwhals-dev/narwhals/discussions/1370)
for an up-to-date plan of future work.

## Used by

Join the party!

- [altair](https://github.com/vega/altair/)
- [hierarchicalforecast](https://github.com/Nixtla/hierarchicalforecast)
- [marimo](https://github.com/marimo-team/marimo)
- [panel-graphic-walker](https://github.com/panel-extensions/panel-graphic-walker)
- [plotly](https://plotly.com)
- [pointblank](https://github.com/posit-dev/pointblank)
- [pymarginaleffects](https://github.com/vincentarelbundock/pymarginaleffects)
- [py-shiny](https://github.com/posit-dev/py-shiny)
- [rio](https://github.com/rio-labs/rio)
- [scikit-lego](https://github.com/koaning/scikit-lego)
- [scikit-playtime](https://github.com/koaning/scikit-playtime)
- [tabmat](https://github.com/Quantco/tabmat)
- [tea-tasting](https://github.com/e10v/tea-tasting)
- [timebasedcv](https://github.com/FBruzzesi/timebasedcv)
- [tubular](https://github.com/lvgig/tubular)
- [vegafusion](https://github.com/vega/vegafusion)
- [wimsey](https://github.com/benrutter/wimsey)

Feel free to add your project to the list if it's missing, and/or
[chat with us on Discord](https://discord.gg/V3PqtB4VA4) if you'd like any support.

## Sponsors and institutional partners

Narwhals is 100% independent, community-driven, and community-owned.
We are extremely grateful to the following organisations for having
provided some funding / development time:

- [Quansight Labs](https://labs.quansight.org)
- [Quansight Futures](https://www.qi.ventures)
- [OpenTeams](https://www.openteams.com)
- [POSSEE initiative](https://possee.org)
- [BYU-Idaho](https://www.byui.edu)

If you contribute to Narwhals on your organization's time, please let us know. We'd be happy to add your employer
to this list!

## Appears on

Narwhals has been featured in several talks, podcasts, and blog posts:

- [Talk Python to me Podcast](https://youtu.be/FSH7BZ0tuE0)
  Ahoy, Narwhals are bridging the data science APIs

- [Python Bytes Podcast](https://www.youtube.com/live/N7w_ESVW40I?si=y-wN1uCsAuJOKlOT&t=382)
  Episode 402, topic #2

- [Super Data Science: ML & AI Podcast](https://www.youtube.com/watch?v=TeG4U8R0U8U)  
  Narwhals: For Pandas-to-Polars DataFrame Compatibility

- [Sample Space Podcast | probabl](https://youtu.be/8hYdq4sWbbQ?si=WG0QP1CZ6gkFf18b)  
  How Narwhals has many end users ... that never use it directly. - Marco Gorelli

- [The Real Python Podcast](https://www.youtube.com/watch?v=w5DFZbFYzCM)
  Narwhals: Expanding DataFrame Compatibility Between Libraries

- [Pycon Lithuania](https://www.youtube.com/watch?v=-mdx7Cn6_6E)  
  Marco Gorelli - DataFrame interoperatiblity - what's been achieved, and what comes next?

- [Pycon Italy](https://www.youtube.com/watch?v=3IqUli9XsmQ)  
  How you can write a dataframe-agnostic library - Marco Gorelli

- [Polars Blog Post](https://pola.rs/posts/lightweight_plotting/)  
  Polars has a new lightweight plotting backend

- [Quansight Labs blog post (w/ Scikit-Lego)](https://labs.quansight.org/blog/scikit-lego-narwhals)  
  How Narwhals and scikit-lego came together to achieve dataframe-agnosticism

## Why "Narwhals"?

[Coz they are so awesome](https://youtu.be/ykwqXuMPsoc?si=A-i8LdR38teYsos4).

Thanks to [Olha Urdeichuk](https://www.fiverr.com/olhaurdeichuk) for the illustration!
