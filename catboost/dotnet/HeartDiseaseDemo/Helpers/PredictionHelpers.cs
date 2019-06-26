using CatBoostNet;
using HeartDiseaseDemo.Models;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;

namespace HeartDiseaseDemo.Helpers
{
    public class PredictionHelpers
    {
        public static double Predict(PatientModel model, CatBoostModel predictor)
        {
            var mlContext = new MLContext();
            var coll = mlContext.Data.LoadFromEnumerable(new List<PatientModel> { model });
            var pred = predictor.Transform(coll);
            double logit = mlContext.Data.CreateEnumerable<CatBoostValuePrediction>(pred, reuseRowObject: false).Single().OutputValues[0];
            return Sigmoid(logit);
        }

        private static double Sigmoid(double logit)
        {
            return 1.0 / (1.0 + Math.Exp(-logit));
        }
    }
}
