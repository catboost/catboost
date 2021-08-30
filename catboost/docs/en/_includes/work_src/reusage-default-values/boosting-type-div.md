
#### {{ calcer_type__cpu }}
{{ fit__boosting-type__plain }}
#### {{ calcer_type__gpu }}

- Any number of objects, {{ error-function--MultiClass }} or {{ error-function--MultiClassOneVsAll }} mode: {{ fit__boosting-type__plain }}
- More than 50 thousand objects, any mode: {{ fit__boosting-type__plain }}
- Less than or equal to 50 thousand objects, any mode but {{ error-function--MultiClass }} or {{ error-function--MultiClassOneVsAll }}: {{ fit__boosting-type__ordered }}
