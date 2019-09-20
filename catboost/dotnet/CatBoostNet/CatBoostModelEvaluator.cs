using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace CatBoostNet
{
    /// <summary>
    /// Low-level API for interacting with CatBoost library.
    /// Should be used only for creating custom predictors —
    /// if you just want to run common classification/regression task,
    /// use <see cref="CatBoostModel"/> instead.
    /// </summary>
    public class CatBoostModelEvaluator
    {
        /// <summary>
        /// Structure for storing minimal model info — path to the model file
        /// and pointer to the model handler
        /// (as returned by <see cref="ModelCalcerCreate"/>).
        /// </summary>
        private struct CatBoostModelContainer
        {
            /// <summary>
            /// Path to the model file.
            /// </summary>
            public string ModelPath { get; }

            /// <summary>
            /// Pointer to the model handler. <seealso cref="ModelCalcerCreate"/>
            /// </summary>
            public IntPtr ModelHandler { get; }

            /// <summary>
            /// Container constructor.
            /// </summary>
            /// <param name="path">Path to the model file</param>
            public CatBoostModelContainer(string path)
            {
                ModelPath = path;
                ModelHandler = ModelCalcerCreate();
                if (!LoadFullModelFromFile(ModelHandler, ModelPath))
                {
                    string msg = "";
                    // TODO Call `GetErrorString` without crashing everything
                    throw new CatBoostException(
                        "An error has occurred in the LoadFullModelFromFile() method in catboostmodel library.\n" +
                        $"Returned error message: {msg}"
                    );
                }
            }
        }

        /// <summary>
        /// Low-level model evaluator constructor.
        /// </summary>
        /// <param name="path">Path to the model file</param>
        public CatBoostModelEvaluator(string path)
        {
            ModelContainer = new CatBoostModelContainer(path);
        }

        /// <summary>
        /// Low-level model evaluator destructor.
        /// </summary>
        ~CatBoostModelEvaluator() => ModelCalcerDelete(ModelContainer.ModelHandler);

        /// <summary>
        /// Number of trees in the model
        /// </summary>
        public uint TreeCount => CatBoostModelEvaluator.GetTreeCount(ModelContainer.ModelHandler);

        /// <summary>
        /// Number of numeric used in the model input
        /// </summary>
        public uint FloatFeaturesCount => CatBoostModelEvaluator.GetFloatFeaturesCount(ModelContainer.ModelHandler);

        /// <summary>
        /// Number of categorical features used in the model input
        /// </summary>
        public uint CatFeaturesCount => CatBoostModelEvaluator.GetCatFeaturesCount(ModelContainer.ModelHandler);

        /// <summary>
        /// Indices of the categorical features in the model input
        /// </summary>
        // TODO Retrieve those using CatBoost API
        public IEnumerable<int> CatFeaturesIndices { get; set; }

        /// <summary>
        /// Evaluates the model on the input batch.
        /// <seealso cref="EvaluateSingle(float[], string[])"/>
        /// </summary>
        /// <param name="floatFeatures">
        /// 2D array of float features.
        /// Should have the same <c>.GetLength(0)</c> as <paramref name="catFeatures"/>
        /// </param>
        /// <param name="catFeatures">
        /// 2D array of categorical features encoded as strings.
        /// Should have the same <c>.GetLength(0)</c> as <paramref name="floatFeatures"/>
        /// </param>
        /// <returns>2D array with model predictions for all samples in the batch</returns>
        public double[,] EvaluateBatch(float[,] floatFeatures, string[,] catFeatures)
        {
            if (floatFeatures.GetLength(0) != catFeatures.GetLength(0))
            {
                if (floatFeatures.GetLength(0) > 0 && catFeatures.GetLength(0) > 0)
                {
                    throw new CatBoostException("Inconsistent EvaluateBatch arguments:" +
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
                        "An error has occurred in the CalcModelPredictionSingle() method in catboostmodel library.\n" +
                        $"Returned error message: {msg}"
                    );
                }
            }
            finally
            {
                // TODO Deallocate
            }
        }

        /// <summary>
        /// Evaluates the model on the single input.
        /// <seealso cref="EvaluateSingle(float[], string[])"/>
        /// </summary>
        /// <param name="floatFeatures">Array of float features</param>
        /// <param name="catFeatures">Array of categorical features</param>
        /// <returns>Array storing model prediction for the input</returns>
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
                    "An error has occurred in the CalcModelPredictionSingle() method in catboostmodel library.\n" +
                    $"Returned error message: {msg}"
                );
            }
        }

        /// <summary>
        /// Path to the model file
        /// </summary>
        public string ModelPath { get; }

        /// <summary>
        /// Container storing the model handler
        /// </summary>
        private CatBoostModelContainer ModelContainer { get; }

        // CatBoost DLL imports

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
