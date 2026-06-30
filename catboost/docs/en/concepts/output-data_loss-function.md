# Metric

#### {{ output--contains }}

{% include [format-common-phrases-format__contains__metrics](../_includes/work_src/reusage-formats/format__contains__metrics.md) %}


#### {{ output--format }}

- The first row describes the data provided in the file.

    Format:

    ```
    iter<\t><loss 1><loss 2><\t>...<\t><loss N>
    ```

- The metric names are expanded by colon-separated numbers if several validation datasets are input. The numbers correspond to the serial number of the input dataset.
- All the rows except the first contain information for the specific iteration of building the tree.

    Format:

    ```
    <tree index><\t><loss 1><loss 2><\t>...<\t><loss N>
    ```


#### {{ dl--example }}

```
iter<\t>{{ error-function--Logit }}<\t>{{ error-function--AUC }}
0<\t>0.6637258841<\t>0.8800403474
1<\t>0.6358649829<\t>0.8898645092
2<\t>0.6118586328<\t>0.8905880184
3<\t>0.5882755767<\t>0.8911104564
4<\t>0.5665035887<\t>0.8933460724
```

The output for three input validation datasets:
```
iter<\t>RMSE<\t>RMSE:1<\t>RMSE:2
0<\t>0.114824346<\t>0.1105841934<\t>0.08683344953
1<\t>0.1136556268<\t>0.1095536596<\t>0.08584400666
2<\t>0.1125784149<\t>0.10852689<\t>0.08494974738
3<\t>0.1114784956<\t>0.1075251632<\t>0.08401943147
4<\t>0.1103751142<\t>0.106557555<\t>0.08312388916
```

