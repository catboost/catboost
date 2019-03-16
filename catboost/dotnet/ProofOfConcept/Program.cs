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

namespace ProofOfConcept
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Running Iris experiment...");
            if (RunIris())
            {
                Console.WriteLine("Iris experiment ok");
            }
            else
            {
                Console.WriteLine("Iris experiment FAILED");
            }

            Console.ReadKey();
        }

        private static bool RunIris()
        {
            var workdir = Path.Combine(System.AppDomain.CurrentDomain.BaseDirectory, "datasets", "iris");
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
            model.CatFeaturesIndices = new Collection<int> { };
            double[,] res = model.EvaluateBatch(data, new string[df.RowCount, 0]);

            string[] targetLabelList = new string[] { "Iris-setosa", "Iris-versicolor", "Iris-virginica" };
            for (int i = 0; i < res.GetLength(0); ++i)
            {
                int argmax = Enumerable.Range(0, res.GetLength(1)).Select(j => Tuple.Create(res[i, j], j)).Max().Item2;
                string predLabel = targetLabelList[argmax];
                if (predLabel == target[i])
                    Console.WriteLine($"Sample {i + 1} / {res.GetLength(0)}, predicted {predLabel}... ok");
                else
                {
                    Console.WriteLine($"Sample {i + 1} / {res.GetLength(0)}, predicted {predLabel}... FAILED (actual = {target[i]})");
                    return false;
                }
            }

            return true;
        }
    }
}

