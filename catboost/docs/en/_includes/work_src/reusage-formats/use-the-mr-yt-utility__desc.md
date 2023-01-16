
Use the `mapreduce-yt` utility to transfer data from the {{ input_dataset_format__native_catboost }} to YT tables:
```
mapreduce-yt -write -dst <destination_yt_table> < <source_dsv_file>
```
