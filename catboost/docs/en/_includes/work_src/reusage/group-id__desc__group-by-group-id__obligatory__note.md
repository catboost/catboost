
{% note warning %}

All objects in the dataset must be grouped by group identifiers if they are present. I.e., the objects with the same group identifier should follow each other in the dataset.

{% cut "Example" %}

For example, let's assume that the dataset consists of documents $d_{1}, d_{2}, d_{3}, d_{4}, d_{5}$. The corresponding groups are $g_{1}, g_{2}, g_{3}, g_{2}, g_{2}$, respectively. The feature vectors for the given documents are $f_{1}, f_{2}, f_{3}, f_{4}, f_{5}$ respectively. Then the dataset can take the following form:

$\begin{pmatrix} d_{2}&g_{2}&f_{2}\\ d_{4}&g_{2}&f_{4}\\ d_{5}&g_{2}&f_{5}\\ d_{3}&g_{3}&f_{3}\\ d_{1}&g_{1}&f_{1} \end{pmatrix}$

The grouped blocks of lines can be input in any order. For example, the following order is equivalent to the previous one:

$\begin{pmatrix} d_{1}&g_{1}&f_{1}\\ d_{3}&g_{3}&f_{3}\\ d_{2}&g_{2}&f_{2}\\ d_{4}&g_{2}&f_{4}\\ d_{5}&g_{2}&f_{5} \end{pmatrix}$

{% endcut %}

{% endnote %}
