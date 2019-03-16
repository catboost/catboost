using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Runtime.InteropServices;

namespace CatBoostNet
{
    public class CatBoostModelEvaluator
    {
        private struct CatBoostModelContainer
        {
            public string ModelPath { get; }
            public IntPtr ModelHandler { get; }

            public CatBoostModelContainer(string path)
            {
                ModelPath = path;
                ModelHandler = ModelCalcerCreate();
                if (!LoadFullModelFromFile(ModelHandler, ModelPath))
                {
                    string msg = "";
                    throw new CatBoostException(
                        "An error has occured in the LoadFullModelFromFile() method in catboostmodel library.\n" +
                        $"Returned error message: {msg}"
                    );
                }
            }
        }

        public CatBoostModelEvaluator(string path)
        {
            ModelContainer = new CatBoostModelContainer(path);
        }

        ~CatBoostModelEvaluator() => ModelCalcerDelete(ModelContainer.ModelHandler);

        public uint TreeCount => CatBoostModelEvaluator.GetTreeCount(ModelContainer.ModelHandler);

        public uint FloatFeaturesCount => CatBoostModelEvaluator.GetFloatFeaturesCount(ModelContainer.ModelHandler);

        public uint CatFeaturesCount => CatBoostModelEvaluator.GetCatFeaturesCount(ModelContainer.ModelHandler);

        // HACK
        public IEnumerable<int> CatFeaturesIndices { get; set; }

        public double[,] EvaluateBatch(float[,] floatFeatures, string[,] catFeatures)
        {   
            if (floatFeatures.GetLength(0) != catFeatures.GetLength(0))
            {
                if (floatFeatures.GetLength(0) > 0 && catFeatures.GetLength(0) > 0)
                {
                    throw new CatBoostException("Incosistent EvaluateBatch arguments:" +
                        $"got {floatFeatures.GetLength(0)} samples for float features " +
                        $"but {catFeatures.GetLength(0)} samples for cat features");
                }
            }
            uint docs = (uint)Math.Max(floatFeatures.GetLength(0), catFeatures.GetLength(0));
            uint dim = GetDimensionsCount(ModelContainer.ModelHandler);

            IntPtr floatFeaturePtr = PointerTools.AllocateToPtr(floatFeatures);
            IntPtr catFeaturePtr = PointerTools.AllocateToPtr(catFeatures);

            try
            {
                uint resultSize = dim * docs;
                double[] results = new double[resultSize];
                bool res = CatBoostModelEvaluator.CalcModelPrediction(
                    ModelContainer.ModelHandler,
                    docs,
                    floatFeaturePtr, (uint)floatFeatures.GetLength(1),
                    catFeaturePtr, (uint)catFeatures.GetLength(1),
                    results, resultSize
                );
                if (res)
                {
                    double[,] resultMatrix = new double[docs, dim];
                    for (int doc = 0; doc < docs; ++doc)
                    {
                        for (int d = 0; d < dim; ++d)
                        {
                            resultMatrix[doc, d] = results[dim * doc + d];
                        }
                    }
                    return resultMatrix;
                }
                else
                {
                    string msg = "";
                    throw new CatBoostException(
                        "An error has occured in the CalcModelPredictionSingle() method in catboostmodel library.\n" +
                        $"Returned error message: {msg}"
                    );
                }
            }
            finally
            {
                // TODO Deallocate
            }
        }

        public double[] EvaluateSingle(float[] floatFeatures, string[] catFeatures)
        {
            uint resultSize = GetDimensionsCount(ModelContainer.ModelHandler);
            double[] results = new double[resultSize];
            bool res = CatBoostModelEvaluator.CalcModelPredictionSingle(
                ModelContainer.ModelHandler,
                floatFeatures, (uint)floatFeatures.Length,
                catFeatures, (uint)catFeatures.Length,
                results, resultSize
            );
            if (res)
            {
                return results;
            }
            else
            {
                string msg = "";
                throw new CatBoostException(
                    "An error has occured in the CalcModelPredictionSingle() method in catboostmodel library.\n" +
                    $"Returned error message: {msg}"
                );
            }
        }

        public string ModelPath { get; }
        private CatBoostModelContainer ModelContainer { get; }

        [DllImport("catboostmodel")]
        private static extern IntPtr ModelCalcerCreate();

        [DllImport("catboostmodel")]
        private static extern string GetErrorString([In] IntPtr handler);

        [DllImport("catboostmodel")]
        private static extern void ModelCalcerDelete([In, Out] IntPtr handler);

        [DllImport("catboostmodel")]
        private static extern bool LoadFullModelFromFile([In, Out] IntPtr modelHandle, string filename);

        [DllImport("catboostmodel")]
        private static extern uint GetTreeCount([In] IntPtr modelHandle);

        [DllImport("catboostmodel")]
        private static extern uint GetDimensionsCount([In] IntPtr modelHandle);

        [DllImport("catboostmodel")]
        private static extern uint GetFloatFeaturesCount([In] IntPtr modelHandle);

        [DllImport("catboostmodel")]
        private static extern uint GetCatFeaturesCount([In] IntPtr modelHandle);

        [DllImport("catboostmodel")]
        private static extern bool CalcModelPrediction(
            [In] IntPtr modelHandle,
            uint docCount,
            [In] IntPtr floatFeatures, uint floatFeaturesSize,
            [In] IntPtr catFeatures, uint catFeaturesSize,
            [Out] double[] result, uint resultSize
        );

        [DllImport("catboostmodel")]
        private static extern bool CalcModelPredictionSingle(
            [In] IntPtr modelHandle,
            float[] floatFeatures, uint floatFeaturesSize,
            string[] catFeatures, uint catFeaturesSize,
            [Out] double[] result, uint resultSize
        );
    }
}
