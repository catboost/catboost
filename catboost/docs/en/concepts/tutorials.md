# Tutorials

{{ product }} is well covered with educational materials for both novice and advanced machine learners and data scientists.

{% if audience == "internal" %}

{% include [bootcamp](../yandex_specific/_includes/bootcamp-task.md) %}

{% endif %}

{% if audience == "internal" %}

#### {{ product-nirvana }} tutorials

- [Operations overview](../yandex_specific/tutorials/tutorials-nirvana__main.md)
- [Input data formats](../yandex_specific/tutorials/tutorials-nirvana__input-data-format.md)
- [Applying on YT](../yandex_specific/tutorials/tutorials-nirvana__apply-on-yt.md)

{% endif %}

#### Video tutorial

This tutorial gives a short introduction to {{ product }} and showcases its' functionality in Jupyter Notebook.

#### Video

{% include [videos-tutorial-on-using-catboost-in-python-div](../_includes/work_src/reusage-common-phrases/tutorial-on-using-catboost-in-python-div.md) %}

#### Getting started tutorials

- [{{ product }} tutorial](https://github.com/catboost/tutorials/blob/master/python_tutorial.ipynb)
- [Solving classification problems with CatBoost](https://github.com/catboost/tutorials/blob/master/classification/classification_tutorial.ipynb)

These Python tutorials show how to start working with {{ product }}.

Perform the following steps to use them:
1. Download the tutorials using one of the following methods:

    - Click the **Download** button on the github page
    - Clone the whole repository using the following command:
    ```bash
    git clone https://github.com/catboost/tutorials
    ```

1. Run Jupyter Notebook in the directory with the required `ipynb` file.

#### {{ product }} on GPU

[This tutorial](https://github.com/catboost/tutorials/blob/master/tools/google_colaboratory_cpu_vs_gpu_tutorial.ipynb) shows how to run {{ product }} on GPU with Google Colaboratory.

#### Tutorials in the {{ product }} repository

The {{ product }} repository contains [several tutorials]({{ github-repo-tutorials }}) on various topics, including but no limited to:

- how to apply the model
- how to use custom losses
- how to train a ranking model
- how to perform hyperparameter search

#### Courses

Check out a free part of the [Introduction to Competitive Data Science](https://stepik.org/a/108888) course. The assignment helps to explore all basic functions and implementation features of the {{ product }} [Python package](python-quickstart.md) and understand how to win a Data Science Competition. (in Russian)


#### Applying {{ product }} models in {{ other-products__clickhouse }}

The {{ other-products__clickhouse }} documentation contains a [tutorial]({{ clickhouse__applying-catboost-tutorial }}) on applying a {{ product }} model in {{ other-products__clickhouse }}.
