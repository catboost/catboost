Mode | How splits are chosen
----- | -----
Median | Include an approximately equal number of objects in every bucket.
Uniform | Generate splits by dividing the `[min_feature_value, max_feature_value]` segment into subsegments of equal length. Absolute values of the feature are used in this case.
UniformAndQuantiles | Combine the splits obtained in the following modes, after first halving the quantization size provided by the starting parameters for each of them:<br/>- Median.<br/>- Uniform.
MaxLogSum | Maximize the value of the following expression inside each bucket:<br/>$\sum\limits_{i=1}^{n}\log(weight){ , where}$<br/>- $n$ — The number of distinct objects in the bucket.<br/>- $weight$ — The number of times an object in the bucket is repeated.
MinEntropy | Minimize the value of the following expression inside each bucket:<br/>$\sum \limits_{i=1}^{n} weight \cdot log (weight) { , where}$<br/>- $n$ — The number of distinct objects in the bucket.<br/>- $weight$ — The number of times an object in the bucket is repeated.
GreedyLogSum | Maximize the greedy approximation of the following expression inside every bucket:<br/>$\sum\limits_{i=1}^{n}\log(weight){ , where}$<br/>- $n$ — The number of distinct objects in the bucket.<br/>- $weight$ — The number of times an object in the bucket is repeated.
