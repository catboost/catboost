
#### {{ loss-functions__params__auc__type__Classic }}
$\displaystyle\frac{ \sum_q \sum_{i, j \in q}  \sum I(a_{i}, a_{j}) \cdot w_{i} \cdot w_{j}} { \sum_q \sum_{i, j \in q}
                      \sum w_{i} \cdot w_{j}}$
The sum is calculated on all pairs of objects $(i,j)$ such that:
- $t_{i} = 0$
- $t_{j} = 1$
- $I(x, y) = \begin{cases} 0{ , } & x < y \\ 0.5{ , } & x=y \\ 1{ , } & x>y \end{cases}$

Refer to the [Wikipedia article]({{ wikipedia_under-the-curve }}) for details.

If the target type is not binary, then every object with target value $t$ and weight $w$ is replaced with two objects for the metric calculation:

- $o_{1}$ with weight $t \cdot w$ and target value 1
- $o_{2}$ with weight $(1 – t) \cdot w$ and target value 0.

Target values must be in the range [0; 1].

#### {{ loss-functions__params__auc__type__Ranking }}

$\displaystyle\frac{ \sum_q \sum_{i, j \in q}  \sum I(a_{i}, a_{j}) \cdot w_{i} \cdot w_{j}} { \sum_q \sum_{i, j \in q} \sum w_{i} * w_{j}}$

The sum is calculated on all pairs of objects $(i,j)$ such that:
- $t_{i} < t_{j}$
- $I(x, y) = \begin{cases} 0{ , } & x < y \\ 0.5{ , } & x=y \\ 1{ , } & x>y \end{cases}$
