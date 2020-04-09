using CatBoostNet;
using Deedle;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net;

namespace CatBoostNetTests {
    [TestClass]
    public class CatBoostModelEvaluatorTest {
        [TestMethod]
        public void RunIrisTest() {
            var workdir = Path.Combine(System.AppDomain.CurrentDomain.BaseDirectory, "testbed", "iris");
            var dsPath = Path.Combine(workdir, "iris.data");
            var df = Frame.ReadCsv(dsPath, hasHeaders: false);
            df.RenameColumns(new Collection<string>
            {
                "sepal length", "sepal width", "petal length", "petal width", "target"
            });
            var target = df.Rows.Select(obj => obj.Value["target"]).Values.Select(x => (string)x).ToArray();
            df.DropColumn("target");
            var data = df.ToArray2D<float>();

            var model = new CatBoostModelEvaluator(Path.Combine(workdir, "iris_model.cbm"));
            model.CatFeaturesIndices = new Collection<int>();
            double[,] res = model.EvaluateBatch(data, new string[df.RowCount, 0]);

            string[] targetLabelList = new string[] { "Iris-setosa", "Iris-versicolor", "Iris-virginica" };
            string errors = "";
            for (int i = 0; i < res.GetLength(0); ++i) {
                int argmax = Enumerable.Range(0, res.GetLength(1)).Select(j => Tuple.Create(res[i, j], j)).Max().Item2;
                string predLabel = targetLabelList[argmax];
                if (predLabel != target[i]) {
                    Console.WriteLine(predLabel);
                    Console.WriteLine(target[i]);
                    errors += $"#{i + 1} ";
                }
            }
            Assert.AreEqual(errors.Length, 0, $"Iris test failed on samples: #{errors}");
        }

        [TestMethod]
        public void RunBostonTest() {
            var workdir = Path.Combine(System.AppDomain.CurrentDomain.BaseDirectory, "testbed", "boston");
            var dsPath = Path.Combine(workdir, "housing.data");

            List<float[]> featureList = new List<float[]>();
            List<double> targetList = new List<double>();
            var pointCulture = new CultureInfo("en") {
                NumberFormat = {
                    NumberDecimalSeparator = "."
                }
            };
            using (TextReader textReader = new StreamReader(dsPath)) {
                while (textReader.Peek() != -1) {
                    var tokens = textReader.ReadLine().Split(' ').Where(x => x != "").ToList();
                    var last = tokens.Last();
                    targetList.Add(double.Parse(last, pointCulture));
                    featureList.Add(tokens.SkipLast(1).Select(x => float.Parse(x, pointCulture)).ToArray());
                }
            }

            if (featureList.Where(x => x.Length != featureList.First().Length).Any()) {
                throw new InvalidDataException("Inconsistent column count in housing.data");
            }

            double[] target = targetList.ToArray();
            float[,] features = new float[featureList.Count, featureList.First().Length];
            for (int i = 0; i < featureList.Count; ++i) {
                for (int j = 0; j < featureList.First().Length; ++j) {
                    features[i, j] = featureList[i][j];
                }
            }

            var model = new CatBoostModelEvaluator(Path.Combine(workdir, "boston_housing_model.cbm"));
            model.CatFeaturesIndices = new Collection<int>();
            double[,] res = model.EvaluateBatch(features, new string[featureList.Count, 0]);

            var deltas = Enumerable.Range(0, featureList.Count).Select(i => new {
                Index = i + 1,
                LogDelta = Math.Abs(res[i, 0] - Math.Log(target[i])),
                Pred = Math.Exp(res[i, 0]),
                Target = target[i]
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
        public void RunMushroomTest() {
            var workdir = Path.Combine(System.AppDomain.CurrentDomain.BaseDirectory, "testbed", "mushrooms");
            var dsPath = Path.Combine(workdir, "mushrooms.csv");

            var df = Frame.ReadCsv(dsPath, hasHeaders: false);
            var target = df.Rows.Select(obj => obj.Value["Column1"]).Values.Select(x => (string)x).ToArray();
            df.DropColumn("Column1");
            var data = df.ToArray2D<string>();

            var model = new CatBoostModelEvaluator(Path.Combine(workdir, "mushroom_model.cbm"));
            model.CatFeaturesIndices = Enumerable.Range(0, df.ColumnCount).ToList();
            double[,] res = model.EvaluateBatch(new float[df.RowCount, 0], data);

            string[] targetLabelList = new string[] { "e", "p" };
            for (int i = 0; i < res.GetLength(0); ++i) {
                int argmax = res[i, 0] > 0 ? 1 : 0;
                string predLabel = targetLabelList[argmax];

                Assert.IsTrue(
                    predLabel == target[i],
                    $"Mushroom test crashed on sample {i + 1}"
                );
            }
        }
    }
}
