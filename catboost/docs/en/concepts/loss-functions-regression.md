# Regression: objectives and metrics

- [Objectives and metrics](#objectives-and-metrics)
- [{{ title__loss-functions__text__optimization }}](#usage-inforimation)

## Objectives and metrics

### {{ error-function--MAE }} {#MAE}

$\displaystyle\frac{\sum\limits_{i=1}^{N} w_{i} | a_{i} - t_{i}| }{\sum\limits_{i=1}^{N} w_{i}}$

**{{ optimization }}** See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}

### {{ error-function--MAPE }} {#MAPE}

$\displaystyle\frac{\sum\limits_{i=1}^{N} w_{i} \displaystyle\frac{|a_{i}- t_{i}|}{\max(1, |t_{i}|)}}{\sum\limits_{i=1}^{N}w_{i}}$

**{{ optimization }}** See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}

### {{ error-function--Poisson }} {#Poisson}

$\displaystyle\frac{\sum\limits_{i=1}^{N} w_{i} \left(e^{a_{i}} - a_{i}t_{i}\right)}{\sum\limits_{i=1}^{N}w_{i}}$

**{{ optimization }}** See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}


### {{ error-function--Quantile }} {#Quantile}

$\displaystyle\frac{\sum\limits_{i=1}^{N} (\alpha - I(t_{i} \leq a_{i}))(t_{i} - a_{i}) w_{i} }{\sum\limits_{i=1}^{N} w_{i}}$

**{{ optimization }}** See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}

{% cut "{{ loss-functions__params__alpha }}" %}

The coefficient used in quantile-based losses.

_Default:_ {{ fit--alpha }}

{% endcut %}

### {{ error-function--MultiQuantile }} {#MultiQuantile}

$\displaystyle\frac{\sum\limits_{i=1}^{N} w_{i} \sum\limits_{q=1}^{Q} (\alpha_{q} - I(t_{i} \leq a_{i,q}))(t_{i} - a_{i,q}) }{\sum\limits_{i=1}^{N} w_{i}}$

**{{ optimization }}** See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}

{% cut "{{ loss-functions__params__alpha }}" %}

The vector of coefficients used in multi-quantile loss.

_Default:_ {{ fit--alpha }}

{% endcut %}

### {{ error-function--RMSE }} {#RMSE}

$\displaystyle\sqrt{\displaystyle\frac{\sum\limits_{i=1}^N (a_{i}-t_{i})^2 w_{i}}{\sum\limits_{i=1}^{N}w_{i}}}$

**{{ optimization }}** See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}

### RMSEWithUncertainty {#RMSEWithUncertainty}

$\displaystyle-\frac{\sum_{i=1}^N w_i \log N(t_{i} \vert a_{i,0}, e^{2a_{i,1}})}{\sum_{i=1}^{N}w_{i}} = \frac{1}{2}\log(2\pi) +\frac{\sum_{i=1}^N w_i\left(a_{i,1} + \frac{1}{2} e^{-2a_{i,1}}(t_i - a_{i, 0})^2 \right)}{\sum_{i=1}^{N}w_{i}}$,
where $t$ is target, a 2-dimensional approx $a_0$ is target predict, $a_1$ is $\log \sigma$ predict, and $N(y\vert \mu,\sigma^2) = \frac{1}{\sqrt{2 \pi\sigma^2}} \exp(-\frac{(y-\mu)^2}{2\sigma^2})$ is the probability density function of the normal distribution.

See the [Uncertainty section](../references/uncertainty.md) for more details.

**{{ optimization }}** See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}

### {{ error-function--LogLinQuantile }} {#LogLinQuantile}

Depends on the condition for the ratio of the label value and the resulting value:
$\begin{cases} \displaystyle\frac{\sum\limits_{i=1}^{N} \alpha |t_{i} - e^{a_{i}} | w_{i}}{\sum\limits_{i=1}^{N} w_{i}} & t_{i} > e^{a_{i}} \\ \displaystyle\frac{\sum\limits_{i=1}^{N} (1 - \alpha) |t_{i} - e^{a_{i}} | w_{i}}{\sum\limits_{i=1}^{N} w_{i}} & t_{i} \leq e^{a_{i}} \end{cases}$

**{{ optimization }}** See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}

{% cut "{{ loss-functions__params__alpha }}" %}

The coefficient used in quantile-based losses.

_Default:_  {{ fit--alpha }}

{% endcut %}

### {{ error-function__lq }} {#lq}

$\displaystyle\frac{\sum\limits_{i=1}^N |a_{i} - t_{i}|^q w_i}{\sum\limits_{i=1}^N w_{i}}$

**{{ optimization }}** See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}

{% cut "{{ loss-functions__params__q }}" %}

The power coefficient.<br/><br/>Valid values are real numbers in the following range:  $[1; +\infty)$

_Default:_ {{ loss-functions__params__q__default }}

{% endcut %}


### {{ error-function__Huber }} {#Huber}

$L(t, a) = \sum\limits_{i=0}^N l(t_i, a_i) \cdot w_{i}}$

$l(t,a) = \begin{cases} \frac{1}{2} (t - a)^{2} { , } & |t -a| \leq \delta \\ \delta|t -a| - \frac{1}{2} \delta^{2} { , } & |t -a| > \delta \end{cases}$

{{ title__loss-functions__text__user-defined-params }}:

{% cut "{{ loss-functions__params__delta }}" %}

The $\delta$ parameter of the {{ error-function__Huber }} metric.

_Default:_ {{ loss-functions__params__q__default }}

{% endcut %}

**{{ optimization }}** See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}


### {{ error-function__Expectile }} {#Expectile}

$\displaystyle\frac{\sum\limits_{i=1}^{N} |\alpha - I(t_{i} \leq a_{i})|(t_{i} - a_{i})^2 w_{i} }{\sum\limits_{i=1}^{N} w_{i}}$

**{{ optimization }}** See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}

{% cut "{{ loss-functions__params__alpha }}" %}

The coefficient used in expectile-based losses.

_Default:_  {{ fit--alpha }}

{% endcut %}

### {{ error-function__Tweedie }} {#Tweedie}

$\displaystyle\frac{\sum\limits_{i=1}^{N}\left(\displaystyle\frac{e^{a_{i}(2-\lambda)}}{2-\lambda} - t_{i}\frac{e^{a_{i}(1-\lambda)}}{1-\lambda} \right)\cdot w_{i}}{\sum\limits_{i=1}^{N} w_{i}}$

$\lambda$ is the value of the {{ loss-functions__params__variance_power }} parameter.

**{{ optimization }}** See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}

{% cut "{{ loss-functions__params__variance_power }}" %}

The variance of the Tweedie distribution.

Supported values are in the range (1;2).

_Default:_ {{ loss-functions__params__q__default }}

{% endcut %}

### {{ error-function__LogCosh }} {#LogCosh}

$\displaystyle\frac{\sum_{i=1}^N w_i \log(\cosh(a_i - t_i))}{\sum_{i=1}^N w_i}$

**{{ optimization }}** See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}


### {{ error-function__FairLoss }} {#FairLoss}

$\displaystyle\frac{\sum\limits_{i=1}^{N} c^2\left(\frac{|t_{i} - a_{i} |}{c} - \log\left(\frac{|t_{i} - a_{i} |}{c} + 1\right)\right)w_{i}}{\sum\limits_{i=1}^{N} w_{i}}$

$c$ is the value of the {{ loss-functions__params__smoothness }} parameter.

**{{ no-optimization }}** See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}

{% cut "{{ loss-functions__params__use_weights }}" %}

The smoothness coefficient. Valid values are real values in the following range $(0; +\infty)$.

_Default:_ {{ fit--smoothness }}

{% endcut %}


### {{ error-function__NumErrors }} {#NumErrors}

The proportion of predictions, for which the difference from the label value exceeds the specified value `{{ loss-function__params__greater-than }}`.

$\displaystyle\frac{\sum\limits_{i=1}^{N} I(|a_{i} - t_{i}|\geq \text{greater\_than}) w_{i}}{\sum\limits_{i=1}^{N} w_{i}}$

{{ title__loss-functions__text__user-defined-params }}: {{ loss-function__params__greater-than }}

**{{ no-optimization }}** See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}

### {{ error-function__SMAPE }} {#SMAPE}

$\displaystyle\frac{100 \sum\limits_{i=1}^{N}\displaystyle\frac{w_{i} |a_{i} - t_{i} |}{(| t_{i} | + | a_{i} |) / 2}}{\sum\limits_{i=1}^{N} w_{i}}$

**{{ no-optimization }}** See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}

### {{ error-function--R2 }} {#R2}

$1 - \displaystyle\frac{\sum\limits_{i=1}^{N} w_{i} (a_{i} - t_{i})^{2}}{\sum\limits_{i=1}^{N} w_{i} (\bar{t} - t_{i})^{2}}$
$\bar{t}$ is the average label value:
$\bar{t} = \frac{1}{N}\sum\limits_{i=1}^{N}t_{i}$

**{{ no-optimization }}** See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}


### {{ error-function__MSLE }} {#MSLE}

$\displaystyle\frac{\sum\limits_{i=1}^{N} w_{i} (\log (1 + t_{i}) - \log (1 + a_{i}))^{2}}{\sum\limits_{i=1}^{N} w_{i}}$

**{{ no-optimization }}** See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}


### {{ error-function__MedianAbsoluteError }} {#MedianAbsoluteError}

$\displaystyle\text{median}(|t_{1} - a_{1}|, ..., |t_{i} - a_{N}|)$

**{{ no-optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

No.

### {{ error-function__Cox }} {#Cox}

$\displaystyle\sum\limits_{t_{i} > 0}\left( a_{i} - \log\sum\limits_{|t_{j}| \ge t_{i}} \exp(a_{j})\right)$

Above, positive labels $t_{i} > 0$ mean occurence of the event.

**{{ optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

No.


## {{ title__loss-functions__text__optimization }} {#usage-information}

| Name                                                            | Optimization            | GPU Support             |
------------------------------------------------------------------|-------------------------|-------------------------|
[{{ error-function--MAE }}](#MAE)                                 |     +                   |     +                   |
[{{ error-function--MAPE }}](#MAPE)                               |     +                   |     +                   |
[{{ error-function--Poisson }}](#Poisson)                         |     +                   |     +                   |
[{{ error-function--Quantile }}](#Quantile)                       |     +                   |     +                   |
[{{ error-function--MultiQuantile }}](#MultiQuantile)             |     +                   |     -                   |
[{{ error-function--RMSE }}](#RMSE)                               |     +                   |     +                   |
[RMSEWithUncertainty](#RMSEWithUncertainty)                       |     +                   |     -                   |
[{{ error-function--LogLinQuantile }}](#LogLinQuantile)           |     +                   |     +                   |
[{{ error-function__lq }}](#lq)                                   |     +                   |     +                   |
[{{ error-function__Huber }}](#Huber)                             |     +                   |     +                   |
[{{ error-function__Expectile }}](#Expectile)                     |     +                   |     +                   |
[{{ error-function__Tweedie }}](#Tweedie)                         |     +                   |     +                   |
[{{ error-function__LogCosh }}](#LogCosh)                         |     +                   |     -                   |
[{{ error-function__Cox }}](#Cox)                                 |     +                   |     -                   |
[{{ error-function__FairLoss }}](#FairLoss)                       |     -                   |     -                   |
[{{ error-function__NumErrors }}](#NumErrors)                     |     -                   |     +                   |
[{{ error-function__SMAPE }}](#SMAPE)                             |     -                   |     -                   |
[{{ error-function--R2 }}](#R2)                                   |     -                   |     -                   |
[{{ error-function__MSLE }}](#MSLE)                               |     -                   |     -                   |
[{{ error-function__MedianAbsoluteError }}](#MedianAbsoluteError) |     -                   |     -                   |
