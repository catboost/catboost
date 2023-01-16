{% cut "{{ loss-functions__params__top }}" %}

The number of top samples in a group that are used to calculate the ranking metric. Top samples are either the samples with the largest approx values or the ones with the lowest target values if approx values are the same.

_Default_: {{ loss-functions__params__top__default }}.

{% endcut %}
