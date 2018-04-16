<img src=http://storage.mds.yandex.net/get-devtools-opensource/250854/catboost-logo.png width=300/>

[Website](https://catboost.yandex) |
[Documentation](https://tech.yandex.com/catboost/doc/dg/concepts/about-docpage/) |
[Installation](https://tech.yandex.com/catboost/doc/dg/concepts/cli-installation-docpage/) |
[Release Notes](https://github.com/catboost/catboost/releases)

[![Build Status](https://travis-ci.org/catboost/catboost.svg?branch=master)](https://travis-ci.org/catboost/catboost)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/catboost/catboost?branch=master&svg=true)](https://ci.appveyor.com/project/exprmntr/catboost-jjlbl)
[![License](https://img.shields.io/badge/license-Apache-blue.svg)](https://github.com/catboost/catboost/blob/master/LICENSE)
[![PyPI version](https://badge.fury.io/py/catboost.svg)](https://badge.fury.io/py/catboost)

CatBoost is a machine learning method based on [gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting) over decision trees.

Main advantages of CatBoost:
  - Superior quality when [compared](https://github.com/catboost/benchmarks/blob/master/README.md) with other GBDT libraries.
  - Best in class [inference](https://tech.yandex.com/catboost/doc/dg/concepts/c-plus-plus-api-docpage/) speed.
  - Support for both [numerical and categorical](https://tech.yandex.com/catboost/doc/dg/concepts/algorithm-main-stages-docpage/) features.
  - Fast GPU and multi-GPU support for training (compiled binaries and python package for learning on one host, build [cmd-line MPI version](https://tech.yandex.com/catboost/doc/dg/concepts/cli-installation-docpage/#multi-node-installation) from source to learn on several GPU machines).
  - Data visualization tools [included](https://tech.yandex.com/catboost/doc/dg/features/visualization-docpage/).

Get Started and Documentation
--------------
All CatBoost documentation is available [here](https://tech.yandex.com/catboost/doc/dg/concepts/about-docpage/).

Install CatBoost by following the guide for the
 * [Python package](https://tech.yandex.com/catboost/doc/dg/concepts/python-installation-docpage/)
 * [R-package](https://tech.yandex.com/catboost/doc/dg/concepts/r-installation-docpage/)
 * [command line](https://tech.yandex.com/catboost/doc/dg/concepts/cli-installation-docpage/)

Next you may want to investigate:
* [Tutorials](https://github.com/catboost/catboost/tree/master/catboost/tutorials)
* Training modes on [CPU](https://tech.yandex.com/catboost/doc/dg/features/training-docpage/#training) and [GPU](https://tech.yandex.com/catboost/doc/dg/features/training-on-gpu-docpage/#training-on-gpu)
* [Cross-validation](https://tech.yandex.com/catboost/doc/dg/features/cross-validation-docpage/#cross-validation)
* [Implemented metrics](https://tech.yandex.com/catboost/doc/dg/features/loss-functions-desc-docpage/#loss-functions-desc)
* [Parameters tuning](https://tech.yandex.com/catboost/doc/dg/concepts/parameter-tuning-docpage/)
* [Feature importance calculation](https://tech.yandex.com/catboost/doc/dg/features/feature-importances-calculation-docpage/#feature-importances-calculation)
* [Regular](https://tech.yandex.com/catboost/doc/dg/features/prediction-docpage/#prediction) and [staged](https://tech.yandex.com/catboost/doc/dg/features/staged-prediction-docpage/#staged-prediction) predictions

Catboost models in production
--------------
If you want to evaluate Catboost model in your application read [model api documentation](https://github.com/catboost/catboost/tree/master/catboost/CatboostModelAPI.md).

Questions and bug reports
--------------
* For reporting bugs please use the [catboost/bugreport](https://github.com/catboost/catboost/issues) page.
* Ask your question about CatBoost on [Stack Overflow](https://stackoverflow.com/questions/tagged/catboost).

Help to Make CatBoost Better
----------------------------
* Check out [help wanted](https://github.com/catboost/catboost/labels/help%20wanted) issues to see what can be improved, or open an issue if you want something.
* Add your stories and experience to [Awesome CatBoost](AWESOME.md).
* To contribute to CatBoost you need to first read CLA text and add to your pull request, that you agree to the terms of the CLA. More information can be found
in [CONTRIBUTING.md](https://github.com/catboost/catboost/blob/master/CONTRIBUTING.md)
* Instructions for contributors can be found [here](https://tech.yandex.com/catboost/doc/dg/concepts/development-and-contributions-docpage/).

News
--------------
Latest news are published on [twitter](https://twitter.com/catboostml).

Reference Paper
-------
Anna Veronika Dorogush, Andrey Gulin, Gleb Gusev, Nikita Kazeev, Liudmila Ostroumova Prokhorenkova, Aleksandr Vorobev ["Fighting biases with dynamic boosting"](https://arxiv.org/abs/1706.09516). arXiv:1706.09516, 2017.

Anna Veronika Dorogush, Vasily Ershov, Andrey Gulin ["CatBoost: gradient boosting with categorical features support"](http://learningsys.org/nips17/assets/papers/paper_11.pdf). Workshop on ML Systems
at NIPS 2017.

License
-------
© YANDEX LLC, 2017-2018. Licensed under the Apache License, Version 2.0. See LICENSE file for more details.