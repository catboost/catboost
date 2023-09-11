# Reference papers

#### [{{ ext__papers____arxiv_org_abs_1706_09516__name }}]({{ ext__papers____arxiv_org_abs_1706_09516 }})

_Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, Andrey Gulin. NeurIPS, 2018_

NeurIPS 2018 paper with explanation of Ordered boosting principles and ordered categorical features statistics.

#### [CatBoost: gradient boosting with categorical features support](http://learningsys.org/nips17/assets/papers/paper_11.pdf)

_Anna Veronika Dorogush, Vasily Ershov, Andrey Gulin. Workshop on ML Systems at NIPS 2017_

A paper explaining the {{ product }} working principles: how it handles categorical features, how it fights overfitting, how GPU training and fast formula applier are implemented.

#### [Minimal Variance Sampling in Stochastic Gradient Boosting](https://arxiv.org/abs/1910.13204)

_Bulat Ibragimov, Gleb Gusev. arXiv:1910.13204_

A paper about Minimal Variance Sampling, which is the default sampling in {{ product }}.

#### [Finding Influential Training Samples for Gradient Boosted Decision Trees](https://arxiv.org/abs/1802.06640)

_Boris Sharchilev, Yury Ustinovsky, Pavel Serdyukov, Maarten de Rijke. arXiv:1802.06640_

A paper explaining several ways of extending the framework for finding influential training samples for a particular case of tree ensemble-based models to non-parametric GBDT ensembles under the assumption that tree structures remain fixed and introducing a general scheme of obtaining further approximations to this method that balance the trade-off between performance and computational complexity.

#### [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)

_Scott Lundberg, Su-In Lee. arXiv:1705.07874_

A paper explaining a unified framework for interpreting predictions, SHAP (SHapley Additive exPlanations).

#### [Consistent feature attribution for tree ensembles](https://arxiv.org/abs/1706.06060)

_Scott M. Lundberg, Su-In Lee. arXiv:1706.06060_

A paper explaining fast exact solutions for SHAP (SHapley Additive exPlanation) values, a unique additive feature attribution method based on conditional expectations that is both consistent and locally accurate.

#### [Winning The Transfer Learning Track of Yahoo!’s Learning To Rank Challenge with YetiRank](http://proceedings.mlr.press/v14/gulin11a.html)

_Andrey Gulin, Igor Kuralenok, Dimitry Pavlov. PMLR 14:63-76_

The theory underlying the {{ error-function__YetiRank }} and {{ error-function__YetiRankPairwise }} modes in {{ product }}.

#### [Which Tricks are Important for Learning to Rank?](https://arxiv.org/abs/2204.01500)

_Ivan Lyzhin, Aleksei Ustimenko, Andrey Gulin, Liudmila Prokhorenkova. arXiv:2204.01500_

A paper comparing previously introduced LambdaMART, YetiRank and StochasticRank and proposing an improvement to the {{ error-function__YetiRank }} approach to allow for optimizing specific ranking loss functions.

#### [Gradient Boosting Performs Gaussian Process Inference](https://arxiv.org/abs/2206.05608)

_Aleksei Ustimenko, Artem Beliakov, Liudmila Prokhorenkova. arXiv:2206.05608_

This paper shows that gradient boosting based on symmetric decision trees can be equivalently reformulated as a kernel method that converges to the solution of a certain Kernel Ridge Regression problem. Thus, authors obtain the convergence to a Gaussian Process' posterior mean, which, in turn, allows them to easily transform gradient boosting into a sampler from the posterior to provide better knowledge uncertainty estimates through Monte-Carlo estimation of the posterior variance. It is shown that the proposed sampler allows for better knowledge uncertainty estimates leading to improved out-of-domain detection.
