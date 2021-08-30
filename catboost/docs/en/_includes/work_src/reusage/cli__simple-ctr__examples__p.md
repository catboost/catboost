
Examples

- ```
    simple_ctr='Borders:{{ ctr-types__TargetBorderCount }}=2'
    ```

Two new features with differing quantization settings are generated. The first one concludes that an object belongs to the positive class when the label value exceeds the first border. The second one concludes that an object belongs to the positive class when the label value exceeds the second border.

    
For example, if the label takes three different values (0, 1, 2), the first border is 0.5 while the second one is 1.5.

- ```
    simple_ctr='Buckets:{{ ctr-types__TargetBorderCount }}=2'
    ```

The number of features depends on the number of different labels. For example, three new features are generated if the label takes three different values (0, 1, 2). In this case, the first one concludes that an object belongs to the positive class when the value of the feature is equal to 0 or belongs to the bucket indexed 0. The second one concludes that an object belongs to the positive class when the value of the feature is equal to 1 or belongs to the bucket indexed 1, and so on.

