<img src=http://storage.mds.yandex.net/get-devtools-opensource/250854/catboost-logo.png width=300/>

[Website](https://catboost.ai) |
[Documentation](https://catboost.ai/docs/) |
[Tutorials](https://catboost.ai/docs/concepts/tutorials.html) |
[Installation](https://catboost.ai/docs/concepts/installation.html) |
[Release Notes](https://github.com/catboost/catboost/releases)

[![GitHub license](https://img.shields.io/github/license/catboost/catboost.svg)](https://github.com/catboost/catboost/blob/master/LICENSE)
[![PyPI version](https://badge.fury.io/py/catboost.svg)](https://badge.fury.io/py/catboost)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/catboost.svg)](https://anaconda.org/conda-forge/catboost)
[![GitHub issues](https://img.shields.io/github/issues/catboost/catboost.svg)](https://github.com/catboost/catboost/issues)
[![Telegram](https://img.shields.io/badge/chat-on%20Telegram-2ba2d9.svg)](https://t.me/catboost_en)
[![Twitter](https://img.shields.io/badge/@CatBoostML--_.svg?style=social&logo=twitter)](https://twitter.com/CatBoostML)

CatBoost is a machine learning method based on [gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting) over decision trees.

Main advantages of CatBoost:
--------------
  - Superior quality when [compared](https://github.com/catboost/benchmarks/blob/master/README.md) with other GBDT libraries on many datasets.
  - Best in class [prediction](https://catboost.ai/docs/concepts/c-plus-plus-api.html) speed.
  - Support for both [numerical and categorical](https://catboost.ai/docs/concepts/algorithm-main-stages.html) features.
  - Fast GPU and multi-GPU support for training out of the box.
  - Visualization tools [included](https://catboost.ai/docs/features/visualization.html).
  - Fast and reproducible distributed training with [Apache Spark](https://catboost.ai/en/docs/concepts/spark-overview) and [CLI](https://catboost.ai/en/docs/concepts/cli-distributed-learning).

Get Started and Documentation
--------------
All CatBoost documentation is available [here](https://catboost.ai/docs/).

Install CatBoost by following the guide for the
 * [Python package](https://catboost.ai/en/docs/concepts/python-installation)
 * [R-package](https://catboost.ai/en/docs/concepts/r-installation)
 * [Сommand line](https://catboost.ai/en/docs/concepts/cli-installation)
 * [Package for Apache Spark](https://catboost.ai/en/docs/concepts/spark-installation)

Next you may want to investigate:
* [Tutorials](https://github.com/catboost/tutorials/#readme)
* [Training modes and metrics](https://catboost.ai/docs/concepts/loss-functions.html)
* [Cross-validation](https://catboost.ai/docs/features/cross-validation.html#cross-validation)
* [Parameters tuning](https://catboost.ai/docs/concepts/parameter-tuning.html)
* [Feature importance calculation](https://catboost.ai/docs/features/feature-importances-calculation.html)
* [Regular](https://catboost.ai/docs/features/prediction.html#prediction) and [staged](https://catboost.ai/docs/features/staged-prediction.html#staged-prediction) predictions
* CatBoost for Apache Spark videos: [Introduction](https://youtu.be/47-mAVms-b8) and [Architecture](https://youtu.be/nrGt5VKZpzc)

If you cannot open documentation in your browser try adding yastatic.net and yastat.net to the list of allowed domains in your privacy badger. 

Catboost models in production
--------------
If you want to evaluate Catboost model in your application read [model api documentation](https://github.com/catboost/catboost/tree/master/catboost/CatboostModelAPI.md).

Questions and bug reports
--------------
* For reporting bugs please use the [catboost/bugreport](https://github.com/catboost/catboost/issues) page.
* Ask a question on [Stack Overflow](https://stackoverflow.com/questions/tagged/catboost) with the catboost tag, we monitor this for new questions.
* Seek prompt advice at [Telegram group](https://t.me/catboost_en) or Russian-speaking [Telegram chat](https://t.me/catboost_ru)

Help to Make CatBoost Better
----------------------------
* Check out [open problems](https://github.com/catboost/catboost/blob/master/open_problems/open_problems.md) and [help wanted issues](https://github.com/catboost/catboost/labels/help%20wanted) to see what can be improved, or open an issue if you want something.
* Add your stories and experience to [Awesome CatBoost](AWESOME.md).
* To contribute to CatBoost you need to first read CLA text and add to your pull request, that you agree to the terms of the CLA. More information can be found
in [CONTRIBUTING.md](https://github.com/catboost/catboost/blob/master/CONTRIBUTING.md)
* Instructions for contributors can be found [here](https://catboost.ai/docs/concepts/development-and-contributions.html).

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
© YANDEX LLC, 2017-2024. Licensed under the Apache License, Version 2.0. See LICENSE file for more details.
