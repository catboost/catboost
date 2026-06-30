using System;
using System.Collections.Generic;


namespace CatBoostNet {
    /// <summary>
    /// Low-level API for interacting with CatBoost library.
    /// Should be used only for creating custom predictors —
    /// if you just want to run common classification/regression task,
    /// use <see cref="CatBoostModel"/> instead.
    /// </summary>
    public class CatBoostModelEvaluator : IDisposable {
        /// <summary>
        /// Structure for storing minimal model info — path to the model file
        /// and pointer to the model handler
        /// (as returned by <see cref="CatboostNativeInterface.ModelCalcerCreate"/>).
        /// </summary>
        private struct CatBoostModelContainer {
            /// <summary>
            /// Path to the model file.
            /// </summary>
            public string ModelPath { get; }

            /// <summary>
            /// Pointer to the model handler. <seealso cref="CatboostNativeInterface.ModelCalcerCreate"/>
            /// </summary>
            public IntPtr ModelHandler { get; }

            /// <summary>
            /// Container constructor.
            /// </summary>
            /// <param name="path">Path to the model file</param>
            public CatBoostModelContainer(string path) {
                ModelPath = path;
                ModelHandler = CatboostNativeInterface.ModelCalcerCreate();
                if (!CatboostNativeInterface.LoadFullModelFromFile(ModelHandler, ModelPath)) {
                    var msg = CatboostNativeInterface.GetErrorStringConst(ModelHandler);
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
        public CatBoostModelEvaluator(string path) {
            ModelContainer = new CatBoostModelContainer(path);
        }

        /// <summary>
        /// Low-level model evaluator destructor.
        /// </summary>
        ~CatBoostModelEvaluator() {
            Dispose(false);
        }

        /// <summary>
        /// Number of trees in the model
        /// </summary>
        public uint TreeCount => CatboostNativeInterface.GetTreeCount(ModelContainer.ModelHandler);

        /// <summary>
        /// Number of numeric used in the model input
        /// </summary>
        public uint FloatFeaturesCount => CatboostNativeInterface.GetFloatFeaturesCount(ModelContainer.ModelHandler);

        /// <summary>
        /// Number of categorical features used in the model input
        /// </summary>
        public uint CatFeaturesCount => CatboostNativeInterface.GetCatFeaturesCount(ModelContainer.ModelHandler);

        /// <summary>
        /// Use CUDA GPU device for model evaluation
        /// </summary>
        public bool EnableGpuEvaluation(int deviceId) {
            return CatboostNativeInterface.EnableGPUEvaluation(ModelContainer.ModelHandler, deviceId);
        }

        /// <summary>
        /// Get model metainfo for some key
        /// </summary>
        public string GetModelInfoValue(string key) {
            return CatboostNativeInterface.GetModelInfoValueConst(ModelContainer.ModelHandler, key, (uint) key.Length);
        }

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
        public double[,] EvaluateBatch(float[,] floatFeatures, string[,] catFeatures) {
            if (floatFeatures.GetLength(0) != catFeatures.GetLength(0)) {
                if (floatFeatures.GetLength(0) > 0 && catFeatures.GetLength(0) > 0) {
                    throw new CatBoostException("Inconsistent EvaluateBatch arguments:" +
                        $"got {floatFeatures.GetLength(0)} samples for float features " +
                        $"but {catFeatures.GetLength(0)} samples for cat features");
                }
            }
            uint docs = (uint)Math.Max(floatFeatures.GetLength(0), catFeatures.GetLength(0));
            uint dim = CatboostNativeInterface.GetDimensionsCount(ModelContainer.ModelHandler);


            uint resultSize = dim * docs;
            double[] results = new double[resultSize];
            bool res = false;
            unsafe {
                fixed (float* floatFeaturesPtr = floatFeatures) {
                    using (var catFeatureHolder = new StringPointerHolder(catFeatures)) {
                        float*[] floatFeaturesBeginPtrs = new float*[docs];
                        for (int i = 0; i < floatFeatures.GetLength(0); ++i) {
                            floatFeaturesBeginPtrs[i] = floatFeaturesPtr + i * floatFeatures.GetLength(1);
                        }
                        fixed (float** fff = floatFeaturesBeginPtrs) {
                            res = CatboostNativeInterface.CalcModelPrediction(
                                ModelContainer.ModelHandler,
                                docs,
                                fff, (uint)floatFeatures.GetLength(1),
                                catFeatureHolder.MainPointer, (uint)catFeatures.GetLength(1),
                                results, resultSize
                            );
                        }
                    }
                }
            }
            if (res) {
                double[,] resultMatrix = new double[docs, dim];
                for (int doc = 0; doc < docs; ++doc) {
                    for (int d = 0; d < dim; ++d) {
                        resultMatrix[doc, d] = results[dim * doc + d];
                    }
                }
                return resultMatrix;
            } else {
                var msg = CatboostNativeInterface.GetErrorStringConst(ModelContainer.ModelHandler);
                throw new CatBoostException(
                    "An error has occurred in the CalcModelPredictionSingle() method in catboostmodel library.\n" +
                    $"Returned error message: {msg}"
                );
            }
        }

        /// <summary>
        /// Evaluates the model on the single input.
        /// <seealso cref="EvaluateSingle(float[], string[])"/>
        /// </summary>
        /// <param name="floatFeatures">Array of float features</param>
        /// <param name="catFeatures">Array of categorical features</param>
        /// <returns>Array storing model prediction for the input</returns>
        public double[] EvaluateSingle(float[] floatFeatures, string[] catFeatures) {
            uint resultSize = CatboostNativeInterface.GetDimensionsCount(ModelContainer.ModelHandler);
            double[] results = new double[resultSize];
            bool res = true;
            unsafe {
                fixed (float* floatFeaturesPtr = floatFeatures) {
                    res = CatboostNativeInterface.CalcModelPredictionSingle(
                       ModelContainer.ModelHandler,
                       floatFeatures, (uint)floatFeatures.Length,
                       catFeatures, (uint)catFeatures.Length,
                       results, resultSize
                   );
                }
            }

            if (res) {
                return results;
            } else {
                var msg = CatboostNativeInterface.GetErrorStringConst(ModelContainer.ModelHandler);
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

        /// <summary>
        /// Do not dispose resources twice
        /// </summary>
        private bool isDisposed = false;

        /// <summary>
        /// Dispose of resources
        /// Suppress finalization
        /// </summary>
        public void Dispose() {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Dispose of unmanaged resources
        /// </summary>
        protected virtual void Dispose(bool fromDisposeMethod) {
            if (!isDisposed) {
                CatboostNativeInterface.ModelCalcerDelete(ModelContainer.ModelHandler);
                isDisposed = true;
            }
        }
    }
}
