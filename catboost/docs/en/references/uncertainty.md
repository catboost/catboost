# Uncertainty

There are two main sources of uncertainty: data uncertainty (also known as aleatoric uncertainty) and knowledge uncertainty (also known as epistemic uncertainty).

Data uncertainty arises due to the inherent complexity of the data, such as additive noise or overlapping classes. Importantly, data uncertainty cannot be reduced by collecting more training data.

Knowledge uncertainty arises when the model is given an input from a region that is either sparsely covered by the training data or far from the training data.

A single model trained with special parameter PosteriorSampling is divided into N several models — virtual ensembles, which return N predicted values when they are applied on documents.

[Get more information](https://towardsdatascience.com/tutorial-uncertainty-estimation-with-catboost-255805ff217e)

## Classification

For a document consider the vector of probabilities predicted by an ensemble of:

- Total uncertainty: $H(\bar p) = -\bar p \ln \bar p-(1-\bar p)\ln(1-\bar p)$ where $\bar p = \frac{\sum_i p_i}{N}$, where H is Entropy.
- Data uncertainty: $\frac{1}{N}\sum H(p_i) = \frac{1}{N}\sum_i -p_i \ln p_i-(1- p_i)\ln(1-
    p_i)$.
- Knowledge uncertainty = Total uncertainty - Data uncertainty.

## Regression

For a document consider the vector of predicted values $a = (a_0,..., a_{N-1})$.

In case when the model was trained with RMSEWithUncertainty loss-function an ensemble also predicts a vector of variances $s = (s_0,..., s_{N-1})$.

- Data uncertainty $\bar s$.
- Knowledge uncertainty $Var(a) = \frac{1}{N}\sum (a_i - \bar a_i)^2$.
- Total uncertainty = Data uncertainty + Knowledge uncertainty.

## Related papers

#### [NGBoost: Natural Gradient Boosting for Probabilistic Prediction (2020)](https://arxiv.org/pdf/1910.03225v1.pdf)

_T. Duan et al._

ICML 2020

#### [Uncertainty in Gradient Boosting via Ensembles (2020)](https://arxiv.org/pdf/2006.10562.pdf)

_A. Ustimenko, L. Prokhorenkova and A. Malinin_

arXiv preprint arXiv:2006.10562
