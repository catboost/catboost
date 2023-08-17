# Bootstrap options

The `bootstrap_type` parameter affects the following important aspects of choosing a split for a tree when building the tree structure:

- **Regularization**

    To prevent overfitting, the weight of each training example is varied over steps of choosing different splits (not over scoring different candidates for one split) or different trees.

- **Speeding up**

    When building a new tree, {{ product }} calculates a score for each of the numerous split candidates. The computation complexity of this procedure is $O(|C|\cdot n)$, where:

    - $|C|$ is the number of numerical features, each providing many split candidates.

    - $n$ is the number of examples.

    Usually, this computation dominates over all other steps of each {{ product }} iteration (see Table 1 in the [{{ ext__papers____arxiv_org_abs_1706_09516__name }}](#catboost-unbiased-boosting-with-categorical-features)). Hence, it seems appealing to speed up this procedure by using only a part of examples for scoring all the split candidates.

Depending on the value of the `bootstrap_type` parameter, these ideas are implemented as described in the list below:

* [{{ fit__bootstrap-type__Bayesian }}](#bayesian)
* [{{ fit__bootstrap-type__Bernoulli }}](#bernoulli)
* [{{ fit__bootstrap-type__MVS }}](#mvs) (supported only on {{ calcer_type__gpu }})
* [{{ fit__bootstrap-type__Poisson }}](#poisson)
* [{{ fit__bootstrap-type__No }}](#no)

## Bootstrap types {#bootstrap-types}

### {{ fit__bootstrap-type__Bayesian }} {#bayesian}

The weight of an example is set to the following value:

$w=a^{t} {, where:}$
- $t$ is defined by the `bagging_temperature` parameter
- $a=-log(\psi)$, where $\psi$ is independently generated from Uniform[0,1] . This is equivalent to generating values $a$ for all the examples according to the Bayesian bootstrap procedure (see D. Rubin “The Bayesian Bootstrap”, 1981, Section 2).

{% note info %}

The {{ fit__bootstrap-type__Bayesian }} bootstrap serves only for the regularization, not for speeding up.

{% endnote %}

**Associated parameters:**

{% cut "bagging_temperature" %}

_{{ title__implementation__cli }}:_ `--bagging-temperature`

{% include [reusage-bagging-temperature__desc__div](../_includes/work_src/reusage/bagging-temperature__desc__div.md) %}

{% endcut %}

{% cut "sampling_unit" %}

_{{ title__implementation__cli }}:_ `--sampling-unit`

{% include [reusage-cli__sampling-unit__desc_div](../_includes/work_src/reusage/cli__sampling-unit__desc_div.md) %}

{% endcut %}

### {{ fit__bootstrap-type__Bernoulli }} {#bernoulli}

Corresponds to Stochastic Gradient Boosting (SGB, refer to the [Stochastic gradient boosting](#stochastic-gradient-boosting) for details). Each example is independently sampled for choosing the current split with the probability defined by the `subsample` parameter. All the sampled examples have equal weights. Though SGB was originally proposed for regularization, it speeds up calculations almost $\left(\frac{1}{subsample}\right)$ times.

**Associated parameters:**

{% cut "subsample" %}

_{{ title__implementation__cli }}:_ `--bagging-temperature`

{% include [reusage-cli__sample-rate__desc__div](../_includes/work_src/reusage/cli__sample-rate__desc__div.md) %}

{% endcut %}

{% cut "sampling_unit" %}

_{{ title__implementation__cli }}:_ `--sampling-unit`

{% include [reusage-cli__sampling-unit__desc_div](../_includes/work_src/reusage/cli__sampling-unit__desc_div.md) %}

{% endcut %}

### {{ fit__bootstrap-type__MVS }} {#mvs}

Supported only on {{ calcer_type__cpu }}.

Implements the importance sampling algorithm called Minimum Variance Sampling (MVS).

Scoring of a split candidate is based on estimating of the expected gradient in each leaf (provided by this candidate), where the gradient $g_{i}$ for the example $i$ is calculated as follows:

$g_{i} = \frac{\partial L(y_{i}, z)}{\partial z}|_ {z=M(i)}$, where
- $L$ is the loss function
- $y_{i}$ is the target of the example $i$
- $M(i)$ is the current model prediction for the example $i$ (see the Algorithm 2 in the [{{ ext__papers____arxiv_org_abs_1706_09516__name }}](#catboost-unbiased-boosting-with-categorical-features)).

For this estimation, {{ fit__bootstrap-type__MVS }} samples the _subsample_ examples $i$ such that the largest values of $|g_i|$ are taken with probability $p_{i}=1$ and each other example $i$ is sampled with probability $\displaystyle\frac{|g_i|}{\mu}$, where $\mu$ is the threshold for considering the gradient to be large if the value is exceeded.

Then, the estimate of the expected gradient is calculated as follows:

$\hat{E\, g} = \frac{\sum_{i:\ sampled\ examples} \frac{g_i}{p_i}}{\sum_{i:\ sampled\ examples} \frac{1}{p_i}} { , where}$
- The numerator is the unbiased estimator of the sum of gradients.
- The denominator is the unbiased estimator of the number of training samples.

#### Theoretical basis {#theoretical-basis}

This algorithm provides the minimum variance estimation of the {{ scorefunction__L2 }} split score for a given expected number of sampled examples:

$s=\sum_{i:\ all\ training\ examples} p_{i}$.

Since the score is a fractional function, it is important to reduce the variance of both the numerator and the denominator. The `mvs_reg` (`--mvs-reg`) hyperparameter affects the weight of the denominator and can be used for balancing between the importance and Bernoulli sampling (setting it to 0 implies importance sampling and to $\infty$ - {{ fit__bootstrap-type__Bernoulli }}).. If it is not set manually, the value is {{ fit__mvs_head_fraction }}.

{{ fit__bootstrap-type__MVS }} can be considered as an improved version of the Gradient-based One-Side Sampling (GOSS, see details in the
[paper](#lightgbm-a-highly-efficient-gradient-boosting-decision-tree)) implemented in LightGBM, which samples a given number of top examples by values $|g_i|$ with the probability 1 and samples other examples with the same fixed probability. Due to the theoretical basis, {{ fit__bootstrap-type__MVS }} provides a lower variance of the estimate $\hat{E\, g}$ than GOSS.

{% note info %}

{{ fit__bootstrap-type__MVS }} may not be the best choice for regularization, since sampled examples and their weights are similar for close iterations.

{% endnote %}

**Associated parameters:**

{% cut "mvs_reg" %}

_{{ title__implementation__cli }}:_ `--mvs-reg`

{% include [reusage-cli__mvs-head-fraction__div](../_includes/work_src/reusage/cli__mvs-head-fraction__div.md) %}

{% endcut %}

{% cut "sampling_unit" %}

_{{ title__implementation__cli }}:_ `--sampling-unit`

{% include [reusage-cli__sampling-unit__desc_div](../_includes/work_src/reusage/cli__sampling-unit__desc_div.md) %}

{% endcut %}

### {{ fit__bootstrap-type__Poisson }} {#poisson}

Refer to the [paper](#estimating-uncertainty-for-massive-data-streams) for details; supported only on {{ calcer_type__gpu }})

The weights of examples are i.i.d. sampled from the Poisson distribution with the parameter $-log(1-subsample)$ providing the expected number of examples with positive weights equal to the `subsample` parameter . If `subsample` is equal to 0.66, this approximates the classical bootstrap (sampling $n$ examples with repetitions).

**Associated parameters:**

{% cut "subsample" %}

_{{ title__implementation__cli }}:_ `--bagging-temperature`

{% include [reusage-cli__sample-rate__desc__div](../_includes/work_src/reusage/cli__sample-rate__desc__div.md) %}

{% endcut %}

{% cut "sampling_unit" %}

_{{ title__implementation__cli }}:_ `--sampling-unit`

{% include [reusage-cli__sampling-unit__desc_div](../_includes/work_src/reusage/cli__sampling-unit__desc_div.md) %}

{% endcut %}

### {{ fit__bootstrap-type__No }} {#no}

All training examples are used with equal weights.

**Associated parameters:**

{% cut "sampling_unit" %}

_{{ title__implementation__cli }}:_ `--sampling-unit`

{% include [reusage-cli__sampling-unit__desc_div](../_includes/work_src/reusage/cli__sampling-unit__desc_div.md) %}

{% endcut %}

## Frequency of resampling and reweighting {#frequency-of-resampling-and-reweighting}

The frequency of resampling and reweighting is defined by the `sampling_frequency` parameter:

- {{ fit__sampling-frequency__PerTree }} — Before constructing each new tree
- {{ fit__sampling-frequency__PerTreeLevel }} — Before choosing each new split of a tree

  {% note info %}

  It is recommended to use {{ fit__bootstrap-type__MVS }} when speeding up is an important issue and regularization is not. It is usually the case when operating large data. For regularization, other options might be more appropriate.

  {% endnote %}

## Related papers

#### [Estimating Uncertainty for Massive Data Streams](https://ai.google/research/pubs/pub43157)

_N. Chamandy, O. Muralidharan, A. Najmi, and S. Naid, 2012_

#### [Stochastic gradient boosting](https://www.researchgate.net/publication/222573328_Stochastic_Gradient_Boosting)

_J. H. Friedman_

Computational Statistics & Data Analysis, 38(4):367–378, 2002

#### [Training Deep Models Faster with Robust, Approximate Importance Sampling](https://papers.nips.cc/paper/7957-training-deep-models-faster-with-robust-approximate-importance-sampling)

_T. B. Johnson and C. Guestrin_

In _Advances in Neural Information Processing Systems_, pages 7276–7286, 2018.

#### [Lightgbm: A highly efficient gradient boosting decision tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)

_G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu._.

In _Advances in Neural Information Processing Systems_, pages 3146–3154, 2017.

#### [{{ ext__papers____arxiv_org_abs_1706_09516__name }}]({{ ext__papers____arxiv_org_abs_1706_09516 }})

_Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, Andrey Gulin. NeurIPS, 2018_

NeurIPS 2018 paper with explanation of Ordered boosting principles and ordered categorical features statistics.
