# Time information

#### {{ output--contains }}

{% include [format-common-phrases-format__contains__time](../_includes/work_src/reusage-formats/format__contains__time.md) %}


#### {{ output--format }}

- The rows are added to the file after choosing the next tree.
    
- Each row contains information related to the corresponding tree.
    
    Format:
    ```
    <tree number><\t><time remaining><\t><time passed>
    ```
    
    - `tree number` is the ID of the tree (numbering starts from zero).
    - `time remaining` is the number of milliseconds remaining until the end of training.
    - `time passed` is the number of milliseconds that has passed since training started.

#### {{ output--example }}

```
0<\t>14380<\t>1597
1<\t>12045<\t>3011
2<\t>10605<\t>4545
3<\t>9255<\t>6170
4<\t>7457<\t>7457
5<\t>5783<\t>8675
6<\t>4255<\t>9928
7<\t>2806<\t>11225
8<\t>1387<\t>12486
```

