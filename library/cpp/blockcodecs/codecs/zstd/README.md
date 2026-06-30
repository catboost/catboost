Zstd codecs
=============

This library registers zstd compression codecs as `zstd_1`, ..., `zstd_22`. 
Fast levels are also registered as `zstd_fast_1`, ..., `zstd_fast_7`.

Measured codec performance on every level. Values below are provided just for reference, exact numbers may vary depending on CPU model and type of data being compressed.

| Codec        | Comp. Ratio | Comp. Speed  (MBps) | Decomp. Speed (MBps) |
|--------------|-------------|---------------------|----------------------| 
| lz4          | 0.5876      | 913                 | 4100                 |
| zstd_fast_7  | 0.5783      | 1066                | 2887                 |
| zstd_fast_6  | 0.5733      | 1050                | 2870                 |
| zstd_fast_5  | 0.5528      | 942                 | 2594                 |
| zstd_fast_4  | 0.5529      | 918                 | 2659                 |
| zstd_fast_3  | 0.5408      | 885                 | 2519                 |
| zstd_fast_2  | 0.5132      | 769                 | 2374                 |
| zstd_fast_1  | 0.5119      | 707                 | 2386                 |
| zstd_1       | 0.4691      | 690                 | 1692                 |
| zstd_2       | 0.4083      | 467                 | 1496                 |
| zstd_3       | 0.3505      | 358                 | 1801                 |
| zstd_4       | 0.3356      | 310                 | 1932                 |
| zstd_5       | 0.3175      | 218                 | 1832                 |
| zstd_6       | 0.3168      | 173                 | 1920                 |
| zstd_7       | 0.3081      | 149                 | 1965                 |
| zstd_8       | 0.3077      | 125                 | 2005                 |
| zstd_9       | 0.272       | 128                 | 2179                 |
| zstd_10      | 0.2693      | 98.5                | 2228                 |
| zstd_11      | 0.2684      | 78.6                | 2185                 |
| zstd_12      | 0.2682      | 71.1                | 2231                 |
| zstd_13      | 0.2687      | 27.8                | 2102                 |
| zstd_14      | 0.2676      | 24                  | 2024                 |
| zstd_15      | 0.2663      | 18.8                | 2225                 |
| zstd_16      | 0.257       | 15.2                | 2093                 |
| zstd_17      | 0.2521      | 12.3                | 2072                 |
| zstd_18      | 0.241       | 9.68                | 1696                 |
| zstd_19      | 0.2395      | 7.97                | 1709                 |
| zstd_20      | 0.2337      | 6.26                | 1520                 |
| zstd_21      | 0.2255      | 5.22                | 1442                 |
| zstd_22      | 0.2037      | 3.88                | 1490                 |
