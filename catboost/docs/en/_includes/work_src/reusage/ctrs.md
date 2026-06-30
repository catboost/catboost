**Type :** Borders

**Formula:**

Calculating ctr for the i-th bucket ($i\in[0; k-1]$):

$ctr_{i} = \frac{countInClass + prior}{totalCount + 1} { , where}$

- `countInClass` is how many times the label value exceeded $i$ for objects with the current categorical feature value. It only counts objects that already have this value calculated (calculations are made in the order of the objects after shuffling).
- `totalCount` is the total number of objects (up to the current one) that have a feature value matching the current one.
- `prior` is a number (constant) defined by the starting parameters.


**Type :** Buckets

**Formula:**

Calculating ctr for the i-th bucket ($i\in[0; k]$, creates $k+1$ features):

$ctr_{i} = \frac{countInClass + prior}{totalCount + 1} { , where}$

- `countInClass` is how many times the label value was equal to $i$ for objects with the current categorical feature value. It only counts objects that already have this value calculated (calculations are made in the order of the objects after shuffling).
- `totalCount` is the total number of objects (up to the current one) that have a feature value matching the current one.
- `prior` is a number (constant) defined by the starting parameters.


**Type :** BinarizedTargetMeanValue

**Formula:**

How ctr is calculated:

$ctr = \frac{countInClass + prior}{totalCount + 1} { , where}$

- `countInClass` is the sum of the label values divided by the maximum label value integer ($k$).
- `totalCount` is the total number of objects that have a feature value matching the current one.
- `prior` is a number (constant) defined by the starting parameters.


**Type :** Counter

**Formula:**
How ctr is calculated for the training dataset:
$ctr = \frac{curCount + prior}{maxCount + 1} { , where}$
- `curCount` is the total number of objects in the training dataset with the current categorical feature value.
- `maxCount` the number of objects in the training dataset with the most frequent feature value.
- `prior` is a number (constant) defined by the starting parameters.

How ctr is calculated for the validation dataset:
$ctr = \frac{curCount + prior}{maxCount + 1} { , where}$
- `curCount` computing principles depend on the chosen calculation method:
    - {{ counter-calculation-method--full }} — The sum of the total number of objects in the training dataset with the current categorical feature value and the number of objects in the validation dataset with the current categorical feature value.
    - {{ counter-calculation-method--static }} — The total number of objects in the training dataset with the current categorical feature value

- `maxCount` is the number of objects with the most frequent feature value in one of the combinations of the following sets depending on the chosen calculation method:
    - {{ counter-calculation-method--full }} — The training and the validation datasets.
    - {{ counter-calculation-method--static }} — The training dataset.

- `prior` is a number (constant) defined by the starting parameters.

{% note info %}

This ctr does not depend on the label value.

{% endnote %}
