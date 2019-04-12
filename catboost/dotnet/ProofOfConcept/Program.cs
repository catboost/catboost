using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.IO;

using Microsoft.ML;

using CatBoostNet;

using Deedle;
using Microsoft.ML.Data;
using Microsoft.Data.DataView;
using System.Text.RegularExpressions;

namespace ProofOfConcept
{
    class Program
    {
        static void Main(string[] args)
        {
            int nSucc = 0, nTot = 0;
            bool res;

            Console.WriteLine("Running Iris experiment...");
            res = RunIris();
            if (res) nSucc++;
            nTot++;
            Console.WriteLine($"Iris experiment {(res ? "ok" : "FAILED")}");

            Console.WriteLine("Running Boston experiment...");
            res = RunBoston();
            if (res) nSucc++;
            nTot++;
            Console.WriteLine($"Boston experiment {(res ? "ok" : "FAILED")}");

            Console.WriteLine("Running Mushroom experiment...");
            res = RunMushroom();
            if (res) nSucc++;
            nTot++;
            Console.WriteLine($"Mushroom experiment {(res ? "ok" : "FAILED")}");

            Console.WriteLine($"Total runs:      {nTot}");
            Console.WriteLine($"Successful runs: {nSucc}");

            Console.ReadKey();
        }

        private static bool RunIris()
        {
            var workdir = Path.Combine(System.AppDomain.CurrentDomain.BaseDirectory, "datasets", "iris");
            var dsPath = Path.Combine(workdir, "iris.data");

            MLContext mlContext = new MLContext();

            var model = new CatBoostModel(
                Path.Combine(workdir, "iris_model.cbm"),
                "IrisType",
                mlContext
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
                if (predLabel == x.IrisType)
                    Console.WriteLine($"Sample {i + 1} / {dataPoints.Count()}, predicted {predLabel}... ok");
                else
                {
                    Console.WriteLine($"Sample {i + 1} / {dataPoints.Count()}, predicted {predLabel}... FAILED (actual = {x.IrisType})");
                    return false;
                }
            }

            return true;
        }

        private static bool RunBoston()
        {
            var workdir = Path.Combine(System.AppDomain.CurrentDomain.BaseDirectory, "datasets", "boston");
            var dsPath = Path.Combine(workdir, "housing.data");

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
                "MedV",
                mlContext
            );

            IDataView dataView = mlContext.Data.LoadFromTextFile<BostonDataPoint>(dsPath, hasHeader: false, separatorChar: ',');
            IEnumerable<BostonDataPoint> dataPoints = mlContext.Data.CreateEnumerable<BostonDataPoint>(dataView, reuseRowObject: false);

            var predictions = model.Transform(dataView);
            IEnumerable<CatBoostValuePrediction> predictionsBatch = mlContext.Data.CreateEnumerable<CatBoostValuePrediction>(predictions, reuseRowObject: false);

            var deltas = dataPoints.Zip(predictionsBatch, Tuple.Create).Select(xy => new
            {
                LogDelta = Math.Abs(xy.Item2.OutputValues[0] - Math.Log(xy.Item1.MedV)),
                Pred = Math.Exp(xy.Item2.OutputValues[0]),
                Target = xy.Item1.MedV
            });

            int i = 0;
            foreach (var delta in deltas)
            {
                ++i;
                Console.WriteLine($"Sample #{i} / {deltas.Count()}, pred = {delta.Pred:0.00}, target = {delta.Target:0.00}");
            }

            int totalErrors = deltas.Where(x => x.LogDelta >= .4).Count();
            if (totalErrors > 7) return false;

            return true;
        }

        private static bool RunMushroom()
        {
            var workdir = Path.Combine(System.AppDomain.CurrentDomain.BaseDirectory, "datasets", "mushrooms");
            var dsPath = Path.Combine(workdir, "mushrooms.csv");

            MLContext mlContext = new MLContext();

            var model = new CatBoostModel(
                Path.Combine(workdir, "mushroom_model.cbm"),
                "Class",
                mlContext
            );

            IDataView dataView = mlContext.Data.LoadFromTextFile<MushroomDataPoint>(dsPath, hasHeader: true, separatorChar: ',');
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

                if (predLabel == x.Class)
                    Console.WriteLine($"Sample {i + 1} / {predictionsBatch.Count()}, predicted {predLabel}... ok");
                else
                {
                    Console.WriteLine($"Sample {i + 1} / {predictionsBatch.Count()}, predicted {predLabel}... FAILED (actual = {x.Class})");
                    return false;
                }

                ++i;
            }

            return true;
        }
    }
}

