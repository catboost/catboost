For inference you can use either prediction methods from full packages:

 - Python package: several variants of `predict*` methods of `CatBoost*` classes.
 - R package: [catboost.predict](https://catboost.ai/docs/en/concepts/r-reference_catboost-predict)
 - CatBoost for Apache Spark: `transform` methods of `CatBoost*Model` classes
 - CLI: [`calc` mode](https://catboost.ai/docs/en/concepts/cli-reference_calc-model)

Or use dedicated inference-only CatBoost libraries for
  - C
  - C++
  - JVM
  - Rust
  - .NET
  - Node.js

Or export to one of the following model formats:
  - CoreML
  - ONNX
  - PMML

Or export models as code in:
  - C++
  - Python

See [the documentation](https://catboost.ai/docs/en/concepts/applying__models).
