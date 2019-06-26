# Apply CatBoost model from Rust
This tutorial consists of two parts:
- first part where we preprocess dataset and train the classifier model.
  This part can be found in [train_model.ipynb](train_model.ipynb).
- second part where we load model into Rust application and then apply it.
  This part presented as a small Cargo project. To run, execute `cargo run --release`.
  If you just want to look at code snippets you can go directly to [src/main.rs](src/main.rs).
