using CatBoostNet;
using Microsoft.ML;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Text;

using CatBoostNetTests.Schemas;
using System.Linq;
using System.Text.RegularExpressions;

namespace CatBoostNetTests
{
    [TestClass]
    public class CatBoostModelTest
    {
        [TestMethod]
        public void RunIrisTest()
        {
            var workdir = Path.Combine(System.AppDomain.CurrentDomain.BaseDirectory, "testbed", "iris");
            var dsPath = Path.Combine(workdir, "iris.data");

            try
            {
                DownloadHelpers.DownloadDataset(
                    "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                    dsPath);
            }
            catch (WebException)
            {
                Assert.Fail("Failed to download Iris dataset");
            }

            MLContext mlContext = new MLContext();

            var model = new CatBoostModel(
                Path.Combine(workdir, "iris_model.cbm"),
                "IrisType"
            );

            IDataView dataView = mlContext.Data.LoadFromTextFile<IrisDataPoint>(dsPath, hasHeader: false, separatorChar: ',');
            IEnumerable<IrisDataPoint> dataPoints = mlContext.Data.CreateEnumerable<IrisDataPoint>(dataView, reuseRowObject: false);

            var predictions = model.Transform(dataView);
            IEnumerable<CatBoostValuePrediction> predictionsBatch = mlContext.Data.CreateEnumerable<CatBoostValuePrediction>(predictions, reuseRowObject: false);

            string[] targetLabelList = new string[] { "Iris-setosa", "Iris-versicolor", "Iris-virginica" };

            int i = 0;
            foreach (var xy in dataPoints.Zip(predictionsBatch, Tuple.Create))
            {
                (var x, var y) = xy;
                int argmax = Enumerable.Range(0, y.OutputValues.Length).Select(j => Tuple.Create(y.OutputValues[j], j)).Max().Item2;
                string predLabel = targetLabelList[argmax];
                Assert.IsTrue(predLabel == x.IrisType, $"Iris test crashed on sample #{i + 1}");
            }
        }

        [TestMethod]
        public void RunBostonTest()
        {
            var workdir = Path.Combine(System.AppDomain.CurrentDomain.BaseDirectory, "testbed", "boston");
            var dsPath = Path.Combine(workdir, "housing.data");

            try
            {
                DownloadHelpers.DownloadDataset(
                    "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
                    dsPath);
            }
            catch (WebException)
            {
                Assert.Fail("Failed to download Boston dataset");
            }

            var newDsPath = Path.Combine(workdir, "housing.csv");
            File.WriteAllText(newDsPath, "");
            File.AppendAllLines(
                newDsPath,
                File.ReadLines(dsPath).Select(x => Regex.Replace(x.Trim(), @"\s+", ","))
            );
            dsPath = newDsPath;

            MLContext mlContext = new MLContext();

            var model = new CatBoostModel(
                Path.Combine(workdir, "boston_housing_model.cbm"),
                "MedV"
            );

            IDataView dataView = mlContext.Data.LoadFromTextFile<BostonDataPoint>(dsPath, hasHeader: false, separatorChar: ',');
            IEnumerable<BostonDataPoint> dataPoints = mlContext.Data.CreateEnumerable<BostonDataPoint>(dataView, reuseRowObject: false);

            var predictions = model.Transform(dataView);
            IEnumerable<CatBoostValuePrediction> predictionsBatch = mlContext.Data.CreateEnumerable<CatBoostValuePrediction>(predictions, reuseRowObject: false);

            var deltas = dataPoints
                .Zip(predictionsBatch, Tuple.Create)
                .Zip(Enumerable.Range(1, predictionsBatch.Count()), Tuple.Create)
                .Select(rec => new
            {
                Index = rec.Item2,
                LogDelta = Math.Abs(rec.Item1.Item2.OutputValues[0] - Math.Log(rec.Item1.Item1.MedV)),
                Pred = Math.Exp(rec.Item1.Item2.OutputValues[0]),
                Target = rec.Item1.Item1.MedV
            });

            int totalErrors = deltas.Where(x => x.LogDelta >= .4).Count();
            Assert.IsTrue(
                totalErrors <= 7,
                $"Boston test crashed: expected <= 7 errors, got {totalErrors} error(s) on samples {{" +
                string.Join(
                    ", ",
                    deltas.Where(x => x.LogDelta >= .4).Take(8).Select(x => x.Index + 1)
                ) + ", ...}"
            );
        }

        [TestMethod]
        public void RunMushroom()
        {
            var workdir = Path.Combine(System.AppDomain.CurrentDomain.BaseDirectory, "testbed", "mushrooms");
            var dsPath = Path.Combine(workdir, "mushrooms.csv");

            try
            {
                DownloadHelpers.DownloadDataset(
                    "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data",
                    dsPath);
            }
            catch (WebException)
            {
                Assert.Fail("Failed to download Mushroom dataset");
            }

            MLContext mlContext = new MLContext();

            var model = new CatBoostModel(
                Path.Combine(workdir, "mushroom_model.cbm"),
                "Class"
            );

            IDataView dataView = mlContext.Data.LoadFromTextFile<MushroomDataPoint>(dsPath, hasHeader: false, separatorChar: ',');
            IEnumerable<MushroomDataPoint> dataPoints = mlContext.Data.CreateEnumerable<MushroomDataPoint>(dataView, reuseRowObject: false);

            var predictions = model.Transform(dataView);
            IEnumerable<CatBoostValuePrediction> predictionsBatch = mlContext.Data.CreateEnumerable<CatBoostValuePrediction>(predictions, reuseRowObject: false);

            string[] targetLabelList = new string[] { "e", "p" };

            int i = 0;
            foreach (var xy in dataPoints.Zip(predictionsBatch, Tuple.Create))
            {
                (var x, var y) = xy;

                int argmax = (y.OutputValues[0] > 0) ? 1 : 0;
                string predLabel = targetLabelList[argmax];

                Assert.IsTrue(
                    predLabel == x.Class,
                    $"Mushroom test crashed on sample {i + 1}"
                );
            }
        }
    }
}
