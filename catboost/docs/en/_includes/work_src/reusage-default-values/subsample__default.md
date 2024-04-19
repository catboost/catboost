The default value depends on the dataset size and the bootstrap type:
- Datasets with less than 100 objects — 1
- Datasets with 100 objects or more:
    - {{ fit__bootstrap-type__Poisson }}, {{ fit__bootstrap-type__Bernoulli }} — 0.66
    - {{ fit__bootstrap-type__MVS }} — 0.8

