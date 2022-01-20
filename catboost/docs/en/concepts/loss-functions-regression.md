# Regression: objectives and metrics

- [Objectives and metrics](#objectives-and-metrics)
- [{{ title__loss-functions__text__optimization }}](#used-for-optimization)

## Objectives and metrics

### {{ error-function--MAE }} {#MAE}

$\frac{\sum\limits_{i=1}^{N} w_{i} | a_{i} - t_{i}| }{\sum\limits_{i=1}^{N} w_{i}}$

{{ title__loss-functions__text__user-defined-params }}:

{% include [use-weights__desc__without__full](../_includes/work_src/reusage-loss-functions/use-weights__desc__without__full.md) %}

_Default:_ {{ loss-functions__use_weights__default }}

### {{ error-function--MAPE }} {#MAPE}

$\displaystyle\frac{\sum\limits_{i=1}^{N} w_{i} \displaystyle\frac{|a_{i}- t_{i}|}{Max(1, |t_{i}|)}}{\sum\limits_{i=1}^{N}w_{i}}$

{{ title__loss-functions__text__user-defined-params }}:

{% include [use-weights__desc__without__full](../_includes/work_src/reusage-loss-functions/use-weights__desc__without__full.md) %}

_Default:_ {{ loss-functions__use_weights__default }}


### {{ error-function--Poisson }} {#Poisson}


$\displaystyle\frac{\sum\limits_{i=1}^{N} w_{i} \left(e^{a_{i}} - a_{i}t_{i}\right)}{\sum\limits_{i=1}^{N}w_{i}}$


{{ title__loss-functions__text__user-defined-params }}:

{% include [use-weights__desc__without__full](../_includes/work_src/reusage-loss-functions/use-weights__desc__without__full.md) %}

_Default:_ {{ loss-functions__use_weights__default }}



### {{ error-function--Quantile }} {#Quantile}


$\displaystyle\frac{\sum\limits_{i=1}^{N} (\alpha - 1(t_{i} \leq a_{i}))(t_{i} - a_{i}) w_{i} }{\sum\limits_{i=1}^{N} w_{i}}$

{{ title__loss-functions__text__user-defined-params }}:


{% include [use-weights__desc__without__full](../_includes/work_src/reusage-loss-functions/use-weights__desc__without__full.md) %}

_Default:_ {{ loss-functions__use_weights__default }}

{% cut "{{ loss-functions__params__alpha }}" %}

The coefficient used in quantile-based losses.

{% endcut %}

_Default:_ {{ fit--alpha }}



### {{ error-function--RMSE }} {#RMSE}

$\displaystyle\sqrt{\displaystyle\frac{\sum\limits_{i=1}^N (a_{i}-t_{i})^2 w_{i}}{\sum\limits_{i=1}^{N}w_{i}}}$

{{ title__loss-functions__text__user-defined-params }}:

{% include [use-weights__desc__without__full](../_includes/work_src/reusage-loss-functions/use-weights__desc__without__full.md) %}

_Default:_ {{ loss-functions__use_weights__default }}



### RMSEWithUncertainty {#RMSEWithUncertainty}


$-\frac{1}{N} \sum_{i=1}^N \log p(t_i \vert a_i) = -\frac{1}{N} \sum_{i=1}^N \log(\frac{1}{2 \pi\sigma^2} \exp(-\frac{(y-\mu)^2}{2\sigma^2})) = C +\frac{1}{N}\sum_{i=1}^N \left(a_{i,1} + \frac{1}{2} \exp(-2 a_{i,1} (t_i - a_{i, 0})^2) \right)$, where t is target, a 2-dimensional approx $a_0$ is target predict, $a_1$ is $\log \sigma$ predict, and $p$ has normal distribution $p(t \vert a) = N(y \vert a_0, e^{2a_1}) = N(y \vert \mu, \sigma^2) = \frac{1}{2 \pi\sigma^2} \exp(-\frac{(y-\mu)^2}{2\sigma^2})$

See the [Uncertainty section](../references/uncertainty.md) for more details.


{{ title__loss-functions__text__user-defined-params }}:

{% include [use-weights__desc__without__full](../_includes/work_src/reusage-loss-functions/use-weights__desc__without__full.md) %}

_Default:_ {{ loss-functions__use_weights__default }}



### {{ error-function--LogLinQuantile }} {#LogLinQuantile}


Depends on the condition for the ratio of the label value and the resulting value:
$\begin{cases} \displaystyle\frac{\sum\limits_{i=1}^{N} \alpha |t_{i} - e^{a_{i}} | w_{i}}{\sum\limits_{i=1}^{N} w_{i}} & t_{i} > e^{a_{i}} \\ \displaystyle\frac{\sum\limits_{i=1}^{N} (1 - \alpha) |t_{i} - e^{a_{i}} | w_{i}}{\sum\limits_{i=1}^{N} w_{i}} & t_{i} \leq e^{a_{i}} \end{cases}$

{{ title__loss-functions__text__user-defined-params }}:

{% include [use-weights__desc__without__full](../_includes/work_src/reusage-loss-functions/use-weights__desc__without__full.md) %}

_Default:_ {{ loss-functions__use_weights__default }}

{% cut "{{ loss-functions__params__alpha }}" %}

The coefficient used in quantile-based losses.

{% endcut %}

_Default:_  {{ fit--alpha }}



### {{ error-function__lq }} {#lq}

$\displaystyle\frac{\sum\limits_{i=1}^N |a_{i} - t_{i}|^q w_i}{\sum\limits_{i=1}^N w_{i}}$

{{ title__loss-functions__text__user-defined-params }}:

{% include [use-weights__desc__without__full](../_includes/work_src/reusage-loss-functions/use-weights__desc__without__full.md) %}

_Default:_ {{ loss-functions__use_weights__default }}

{% cut "{{ loss-functions__params__q }}" %}

The power coefficient.<br/><br/>Valid values are real numbers in the following range:  $[1; +\infty)$

{% endcut %}

_Default:_ {{ loss-functions__params__q__default }}


### {{ error-function__Huber }} {#Huber}


$L(t, a) = \sum\limits_{i=0}^N l(t_i, a_i) \cdot w_{i} { , where}$

$l(t,a) = \begin{cases} \frac{1}{2} (t - a)^{2} { , } & |t -a| \leq \delta \\ \delta|t -a| - \frac{1}{2} \delta^{2} { , } & |t -a| > \delta \end{cases}$

{{ title__loss-functions__text__user-defined-params }}:

{% cut "{{ loss-functions__params__delta }}" %}

The $\delta$ parameter of the {{ error-function__Huber }} metric.

{% endcut %}

_Default:_ {{ loss-functions__params__q__default }}

{% include [use-weights__desc__without__full](../_includes/work_src/reusage-loss-functions/use-weights__desc__without__full.md) %}

_Default:_ {{ loss-functions__use_weights__default }}



### {{ error-function__Expectile }} {#Expectile}


$\displaystyle\frac{\sum\limits_{i=1}^{N} |\alpha - 1(t_{i} \leq a_{i})|(t_{i} - a_{i})^2 w_{i} }{\sum\limits_{i=1}^{N} w_{i}}$

{{ title__loss-functions__text__user-defined-params }}:

{% include [use-weights__desc__without__full](../_includes/work_src/reusage-loss-functions/use-weights__desc__without__full.md) %}

_Default:_ {{ loss-functions__use_weights__default }}

{% cut "{{ loss-functions__params__alpha }}" %}

The coefficient used in expectile-based losses.

{% endcut %}

_Default:_  {{ fit--alpha }}


### {{ error-function__Tweedie }} {#Tweedie}

$\displaystyle\frac{\sum\limits_{i=1}^{N}\left(\displaystyle\frac{e^{a_{i}(2-\lambda)}}{2-\lambda} - t_{i}\frac{e^{a_{i}(1-\lambda)}}{1-\lambda} \right)\cdot w_{i}}{\sum\limits_{i=1}^{N} w_{i}} { , where}$

$\lambda$ is the value of the {{ loss-functions__params__variance_power }} parameter.

{{ title__loss-functions__text__user-defined-params }}:

{% include [use-weights__desc__without__full](../_includes/work_src/reusage-loss-functions/use-weights__desc__without__full.md) %}

_Default:_ {{ loss-functions__use_weights__default }}

{% cut "{{ loss-functions__params__variance_power }}" %}

The variance of the Tweedie distribution.

Supported values are in the range (1;2).

{% endcut %}

_Default:_ {{ loss-functions__params__q__default }}



### {{ error-function__FairLoss }} {#FairLoss}

$\displaystyle\frac{\sum\limits_{i=1}^{N} c^2(\frac{|t_{i} - a_{i} |}{c} - \ln(\frac{|t_{i} - a_{i} |}{c} + 1))w_{i}}{\sum\limits_{i=1}^{N} w_{i}} { , where}$

$c$ is the value of the {{ loss-functions__params__smoothness }} parameter.

{{ title__loss-functions__text__user-defined-params }}:

{% include [use-weights__desc__without__full](../_includes/work_src/reusage-loss-functions/use-weights__desc__without__full.md) %}

_Default:_ {{ loss-functions__use_weights__default }}

{% cut "{{ loss-functions__params__use_weights }}" %}

The smoothness coefficient. Valid values are real values in the following range $(0; +\infty)$.

{% endcut %}

_Default:_ {{ fit--smoothness }}



### {{ error-function__NumErrors }} {#NumErrors}

The proportion of predictions, for which the difference from the label value exceeds the specified value `{{ loss-function__params__greater-than }}`.

$\displaystyle\frac{\sum\limits_{i=1}^{N} I\{x\} w_{i}}{\sum\limits_{i=1}^{N} w_{i}} { , where}$

$I\{x\} = \begin{cases} 1 { , } & |a_{i} - t_{i}| > greater\_than \\ 0 { , } & |a_{i} - t_{i}| \leq greater\_than \end{cases}$

{{ title__loss-functions__text__user-defined-params }}: {{ loss-function__params__greater-than }}

Increase the numerator of the formula if the following inequality is met:<br/><br/>$|prediction - label|>value$

{% include [use-weights__desc__without__full](../_includes/work_src/reusage-loss-functions/use-weights__desc__without__full.md) %}

_Default:_ {{ loss-functions__use_weights__default }}






### {{ error-function__SMAPE }} {#SMAPE}


$\displaystyle\frac{100 \sum\limits_{i=1}^{N}\displaystyle\frac{w_{i} |a_{i} - t_{i} |}{(| t_{i} | + | a_{i} |) / 2}}{\sum\limits_{i=1}^{N} w_{i}}$


{{ title__loss-functions__text__user-defined-params }}:

{% include [use-weights__desc__without__full](../_includes/work_src/reusage-loss-functions/use-weights__desc__without__full.md) %}

_Default:_ {{ loss-functions__use_weights__default }}




### {{ error-function--R2 }} {#R2}

$1 - \displaystyle\frac{\sum\limits_{i=1}^{N} w_{i} (a_{i} - t_{i})^{2}}{\sum\limits_{i=1}^{N} w_{i} (\bar{t} - t_{i})^{2}}$
$\bar{t}$ is the average label value:
$\bar{t} = \frac{1}{N}\sum\limits_{i=1}^{N}t_{i}$

{{ title__loss-functions__text__user-defined-params }}:

{% include [use-weights__desc__without__full](../_includes/work_src/reusage-loss-functions/use-weights__desc__without__full.md) %}

_Default:_ {{ loss-functions__use_weights__default }}



### {{ error-function__MSLE }} {#MSLE}

$\displaystyle\frac{\sum\limits_{i=1}^{N} w_{i} (\log_{e} (1 + t_{i}) - \log_{e} (1 + a_{i}))^{2}}{\sum\limits_{i=1}^{N} w_{i}}$

{{ title__loss-functions__text__optimization }}: –

{{ title__loss-functions__text__user-defined-params }}:

{% include [use-weights__desc__without__full](../_includes/work_src/reusage-loss-functions/use-weights__desc__without__full.md) %}


_Default:_ {{ loss-functions__use_weights__default }}

### {{ error-function__MedianAbsoluteError }} {#MedianAbsoluteError}

$median(|t_{1} - a_{1}|, ..., |t_{i} - a_{i}|)$

{{ title__loss-functions__text__optimization }}: –

{{ title__loss-functions__text__user-defined-params }}: –


## {{ title__loss-functions__text__optimization }}

| Name                                                            | Optimization            |
------------------------------------------------------------------|-------------------------|
[{{ error-function--MAE }}](#MAE)                                 |     +                   |
[{{ error-function--MAPE }}](#MAPE)                               |     +                   |
[{{ error-function--Poisson }}](#Poisson)                         |     +                   |
[{{ error-function--Quantile }}](#Quantile)                       |     +                   |
[{{ error-function--RMSE }}](#RMSE)                               |     +                   |
[RMSEWithUncertainty](#RMSEWithUncertainty)                       |     +                   |
[{{ error-function--LogLinQuantile }}](#LogLinQuantile)           |     +                   |
[{{ error-function__lq }}](#lq)                                   |     +                   |
[{{ error-function__Huber }}](#Huber)                             |     +                   |
[{{ error-function__Expectile }}](#Expectile)                     |     +                   |
[{{ error-function__Tweedie }}](#Tweedie)                         |     +                   |
[{{ error-function__FairLoss }}](#FairLoss)                       |     -                   |
[{{ error-function__NumErrors }}](#NumErrors)                     |     -                   |
[{{ error-function__SMAPE }}](#SMAPE)                             |     -                   |
[{{ error-function--R2 }}](#R2)                                   |     -                   |
[{{ error-function__MSLE }}](#MSLE)                               |     -                   |
[{{ error-function__MedianAbsoluteError }}](#MedianAbsoluteError) |     -                   |
