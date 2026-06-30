# Score functions

The common approach to solve supervised learning tasks is to minimize the loss function $L$:

$L\left(f(x), y\right) = \sum\limits_{i} w_{i} \cdot l \left(f(x_{i}), y_{i}\right) + J(f){ , where}$

- $l\left( f(x), y\right)$ is the value of the loss function at the point $(x, y)$
- $w_{i}$ is the weight of the $i$-th object
- $J(f)$ is the regularization term.

For example, these formulas take the following form for linear regression:
- $l\left( f(x), y\right) = w_{i} \left( (\theta, x)  - y \right)^{2}$ (mean squared error)
- $J(f) = \lambda \left| | \theta | \right|_{l2}$ (L2 regularization)


## Gradient boosting {#gradient-boosting}

Boosting is a method which builds a prediction model $F^{T}$ as an ensemble of weak learners $F^{T} = \sum\limits_{t=1}^{T} f^{t}$.

In our case, $f^{t}$ is a decision tree. Trees are built sequentially and each next tree is built to approximate negative gradients $g_{i}$ of the loss function $l$ at predictions of the current ensemble:
$g_{i} = -\frac{\partial l(a, y_{i})}{\partial a} \Bigr|_{a = F^{T-1}(x_{i})}$
Thus, it performs a gradient descent optimization of the function $L$. The quality of the gradient approximation is measured by a score function $Score(a, g) = S(a, g)$.


## Types of score functions {#types-of-score-functions}

Let's suppose that it is required to add a new tree to the ensemble. A score function is required in order to choose between candidate trees. Given a candidate tree $f$ let $a_{i}$ denote $f(x_{i})$, $w_{i}$ — the weight of $i$-th object, and $g_{i}$ – the corresponding gradient of $l$. Let’s consider the following score functions:
- $L2 = - \sum\limits_{i} w_{i} \cdot (a_{i} - g_{i})^{2}$
- $Cosine = \displaystyle\frac{\sum w_{i} \cdot a_{i} \cdot g_{i}}{\sqrt{\sum w_{i}a_{i}^{2}} \cdot \sqrt{\sum w_{i}g_{i}^{2}}}$


## Finding the optimal tree structure {#optimal-tree-structure}

Let's suppose that it is required to find the structure for the tree $f$ of depth 1. The structure of such tree is determined by the index $j$ of some feature and a border value $c$. Let $x_{i, j}$ be the value of the $j$-th feature on the $i$-th object and $a_{left}$ and $a_{right}$ be the values at leafs of $f$. Then, $f(x_{i})$ equals to $a_{left}$ if $x_{i,j} \leq c$ and $a_{right}$ if $x_{i,j} > c$. Now the goal is to find the best $j$ and $c$ in terms of the chosen score function.

For the {{ scorefunction__L2 }} score function the formula takes the following form:

$S(a, g) = -\sum\limits_{i} w_{i} (a_{i} - g_{i})^{2} = - \left( \displaystyle\sum\limits_{i:x_{i,j}\leq c} w_{i}(a_{left} - g_{i})^{2} + \sum\limits_{i: x_{i,j}>c} w_{i}(a_{right} - g_{i})^{2} \right)$

Let's denote $W_{left} = \displaystyle\sum_{i: x_{I,j} \leq c} w_{i}$ and $W_{right} = \displaystyle\sum_{i: x_{i,j} >c} w_{i}$.

The optimal values for $a_{left}$ and $a_{right}$ are the weighted averages:
- $a^{*}_{left} =\displaystyle\frac{\sum\limits_{i: x_{i,j} \leq c} w_{i} g_{i}}{W_{left}}$
- $a^{*}_{right} =\displaystyle\frac{\sum\limits_{i: x_{i,j} > c} w_{i} g_{i}}{W_{right}}$

After expanding brackets and removing terms, which are constant in the optimization:

$j^{*}, c^{*} = argmax_{j, c} W_{left} \cdot (a^{*}_{left})^{2} + W_{right} \cdot (a^{*}_{right})^{2}$

The latter argmax can be calculated by brute force search.

The situation is slightly more complex when the tree depth is bigger than 1:
- {{ scorefunction__L2 }} score function: S is converted into a sum over leaves $S(a,g) = \sum_{leaf} S(a_{leaf}, g_{leaf})$. The next step is to find $j*, c* = argmax_{j,c}{S(\bar a, g)}$, where $\bar a$ are the optimal values in leaves after the $j*, c*$ split.
- {{ growing_policy__Depthwise }} and {{ growing_policy__Lossguide }} methods: $j, c$ are sets of $\{j_k\}, \{c_k\}$. $k$ stands for the index of the leaf, therefore the score function $S$ takes the following form: $S(\bar a, g) = \sum_{l = leaf}S(\bar a(j_l, c_l), g_l)$. Since $S(leaf)$ is a convex function, different $j_{k1}, c_{k1}$ and $j_{k2}, c_{k2}$ (splits for different leaves) can be searched separately by finding the optimal $j*, c* = argmax_{j,c}\{S(leaf_{left}) + S(leaf_{right}) - S(leaf_{before\_split})\}$.
- {{ growing_policy__SymmetricTree }} method: The same $j, c$ are attempted to be found for each leaf, thus it's required to optimize the total sum over all leaves $S(a,g) = \sum_{leaf} S(leaf)$.


## Second-order leaf estimation method {#second-order-functions}

Let's apply the Taylor expansion to the loss function at the point $a^{t-1} = F^{t-1}(x)$:

$L(a^{t-1}_{i} + \phi , y) \approx \displaystyle\sum w_{i} \left[ l_{i} + l^{'}_{i} \phi + \frac{1}{2} l^{''}_{i} \phi^{2} \right] + \frac{1}{2} \lambda ||\phi||_{2}{ , where:}$

- $l_{i} = l(a^{t-1}_{i}, y_{i})$
- $l'_{i} = -\frac{\partial l(a, y_{i})}{\partial a}\Bigr|_{a=a^{t-1}_{i}}$
- $l''_{i} = -\frac{\partial^{2} l(a, y_{i})}{\partial a^{2}}\displaystyle\Bigr|_{a=a^{t-1}_{i}}$
- $\lambda$ is the l2 regularization parameter

Since the first term is constant in optimization, the formula takes the following form after regrouping by leaves:

$\sum\limits_{leaf=1}^{L} \left( \sum\limits_{i \in leaf} w_{i} \left[ l_{i} + l^{'}_{i} \phi_{leaf} + \frac{1}{2} l^{''}_{i} \phi^{2} \right] + \frac{1}{2} \lambda \phi_{leaf}^{2} \right) \to min$

Then let's minimize this expression for each leaf independently:

$\sum\limits_{i \in leaf} w_{i} \left[ l_{i} + l^{'}_{i} \phi_{leaf} + \frac{1}{2} l^{''}_{i} \phi^{2}_{leaf} \right] + \frac{1}{2} \lambda \phi_{leaf}^2 \to min$

Differentiate by leaf value $\phi_{leaf}$:

$\sum\limits_{i \in leaf} w_{i} \left[ l^{'}_{i} + l^{''}_{i} \phi_{leaf} \right] + \lambda \phi_{leaf} = 0$

So, the optimal value of $\phi_{leaf}$ is:

$- \displaystyle\frac{\sum_{i}w_{i}l^{'}_{i}}{\sum_{i}w_{i}l^{''}_{i}+\lambda}$

The summation is over $i$ such that the object $x_{i}$ gets to the considered leaf. Then these optimal values of $\phi_{leaf}$ can be used instead of weighted averages of gradients ($a^{*}_{left}$ and $a^{*}_{right}$ in the example above) in the same score functions.


## {{ product }} score functions {#score-functions}

{{ product }} provides the following score functions:
**Score function:** {{ scorefunction__L2 }}

#### Description


Use the first derivatives during the calculation.


**Score function:** {{ scorefunction__Correlation }} (can not be used with the {{ growing_policy__Lossguide }} tree growing policy)

**Score function:** {{ scorefunction__NewtonL2 }}

#### Description


Use the second derivatives during the calculation. This may improve the resulting quality of the model.


**Score function:** {{ scorefunction__NewtonCorrelation }} (can not be used with the {{ growing_policy__Lossguide }} tree growing policy)


## Per-object and per-feature penalties {#per-object-feature-penalties}

{{ product }} provides the following methods to affect the score with penalties:
- {% include [reusage-cli__first-feature-use-penalties__intro__p](../_includes/work_src/reusage/cli__first-feature-use-penalties__intro__p.md) %}

- {% include [reusage-per-object-feature-penalties__p](../_includes/work_src/reusage/per-object-feature-penalties__p.md) %}

The final score is calculated as follows:
$Score' = Score \cdot \prod_{f\in S}W_{f} - \sum_{f\in S}P_{f} \cdot U(f) - \sum_{f\in S}\sum_{x \in L}EP_{f} \cdot U(f, x)$
- $W_{f}$  is the feature weight
- $P_{f}$  is the per-feature penalty
- $EP_{f}$ is the per-object penalty
- $S$ is the current split
- $L$ is the current leaf
- $U(f) = \begin{cases} 0,& \text{if } f \text{ was used in model already}\\ 1,& \text{otherwise} \end{cases}$
- $U(f, x) = \begin{cases} 0,& \text{if } f \text{ was used already for object } x\\ 1,& \text{otherwise} \end{cases}$


## Usage {#usage}

Use the corresponding parameter to set the score function during the training:

{% note alert %}

{% include [reusage-cli__score-function__processing-unit-type__div](../_includes/work_src/reusage/cli__score-function__processing-unit-type__div.md) %}

{% endnote %}

**{{ python-package }}:** `score_function`

**{{ r-package }}:** `score_function`

**Command-line interface:** `--score-function`

#### Description


The [score type](../concepts/algorithm-score-functions.md) used to select the next split during the tree construction.

Possible values:

- {{ scorefunction__Correlation }} (do not use this score type with the {{ growing_policy__Lossguide }} tree growing policy)
- {{ scorefunction__L2 }}
- {{ scorefunction__NewtonCorrelation }} (do not use this score type with the {{ growing_policy__Lossguide }} tree growing policy)
- {{ scorefunction__NewtonL2 }}


