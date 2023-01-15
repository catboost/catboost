### Overview
Reservoir sampling is a family of randomized algorithms for choosing a simple random sample, without replacement, of k items from a population of unknown size n in a single pass over the items.

### Example

```jupyter
In [1]: from library.python import reservoir_sampling

In [2]: reservoir_sampling.reservoir_sampling(data=range(100), nsamples=10)
Out[2]: [27, 19, 81, 45, 89, 78, 13, 36, 29, 9]
```
