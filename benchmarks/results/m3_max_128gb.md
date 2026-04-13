# CatBoost-MLX vs CatBoost CPU Benchmark

- **Date**: 2026-04-12 21:12:16
- **Hardware**: M3 Max 128GB
- **Python**: 3.13.2
- **catboost**: 1.2.10
- **catboost_mlx**: 0.3.0
- **Iterations**: 100
- **Hyperparameters**: depth=6, learning_rate=0.1, l2_leaf_reg=3.0, random_seed=42

| Dataset   | Loss       | Backend | Time (s) | Iter/s  | Train Loss |
|-----------|------------|---------|----------|---------|------------|
| 10k x 50  | RMSE       | CPU     |     0.20 |   506.8 |     2.7405 |
| 10k x 50  | RMSE       | MLX     |    32.73 |     3.1 |     3.0407 |
| 10k x 50  | Logloss    | CPU     |     0.30 |   332.5 |     0.3227 |
| 10k x 50  | Logloss    | MLX     |    32.08 |     3.1 |     0.4099 |
| 10k x 50  | MultiClass | CPU     |     0.36 |   275.6 |     0.5443 |
| 10k x 50  | MultiClass | MLX     |    64.06 |     1.6 |     0.6337 |
| 100k x 50 | RMSE       | CPU     |     0.41 |   244.3 |     2.7898 |
| 100k x 50 | RMSE       | MLX     |    70.43 |     1.4 |     3.4215 |
| 100k x 50 | Logloss    | CPU     |     0.70 |   142.7 |     0.3502 |
| 100k x 50 | Logloss    | MLX     |    69.55 |     1.4 |     0.4413 |
| 100k x 50 | MultiClass | CPU     |     0.90 |   110.8 |     0.5881 |
| 100k x 50 | MultiClass | MLX     |   138.39 |     0.7 |     0.6662 |
| 500k x 50 | RMSE       | CPU     |     1.18 |    84.5 |     2.9383 |
| 500k x 50 | RMSE       | MLX     |   175.97 |     0.6 |     2.7722 |
| 500k x 50 | Logloss    | CPU     |     1.73 |    57.9 |     0.3656 |
| 500k x 50 | Logloss    | MLX     |   173.38 |     0.6 |     0.4283 |
| 500k x 50 | MultiClass | CPU     |     3.36 |    29.7 |     0.5930 |
| 500k x 50 | MultiClass | MLX     |   346.20 |     0.3 |     0.6594 |
