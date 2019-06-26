Demo project — UCI heart disease
===

Here you can find the demo project featuring CatBoost model inference API.

Demo problem
---

To showcase CatBoost inference API, we use [UCI Heart Disease dataset](https://www.kaggle.com/ronitf/heart-disease-uci) classification problem. The pre-trained model can be found in the `predictor.cbm` file.

How to run
---

If you can't find `CatBoostNet` in the NuGet gallery, then first follow instructions from the `CatBoostNet` project README.

After that, you can just `dotnet run` in this directory and navigate to the `http://localhost:5000`.