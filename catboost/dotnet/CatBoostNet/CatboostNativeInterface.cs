using System;
using System.Runtime.InteropServices;

namespace CatBoostNet {
    /// <summary>
    /// Static class containing definitions for imported functions
    /// </summary>
    internal static class CatboostNativeInterface {
        public const string DynamicLibraryName = "catboostmodel";

        /// <summary>
        /// Create native model handle
        /// </summary>
        /// <returns>Handle (pointer) to empty model</returns>
        [DllImport(DynamicLibraryName)]
        public static extern IntPtr ModelCalcerCreate();

        /// <summary>
        /// Get error string from model library
        /// </summary>
        /// <param name="handler"></param>
        /// <returns>String with textual error description</returns>
        [DllImport(DynamicLibraryName)]
        public static extern string GetErrorString([In] IntPtr handler);

        /// <summary>
        /// Deallocate model by given handle
        /// </summary>
        /// <param name="handler"></param>
        [DllImport(DynamicLibraryName)]
        public static extern void ModelCalcerDelete([In, Out] IntPtr handler);

        [DllImport(DynamicLibraryName)]
        public static extern bool LoadFullModelFromFile([In, Out] IntPtr modelHandle, string filename);

        [DllImport(DynamicLibraryName)]
        public static extern uint GetTreeCount([In] IntPtr modelHandle);

        [DllImport(DynamicLibraryName)]
        public static extern uint GetDimensionsCount([In] IntPtr modelHandle);

        [DllImport(DynamicLibraryName)]
        public static extern uint GetFloatFeaturesCount([In] IntPtr modelHandle);

        [DllImport(DynamicLibraryName)]
        public static extern uint GetCatFeaturesCount([In] IntPtr modelHandle);

        [DllImport(DynamicLibraryName)]
        public unsafe static extern bool CalcModelPrediction(
            [In] IntPtr modelHandle,
            uint docCount,
            [In] float** floatFeatures, uint floatFeaturesSize,
            [In] IntPtr catFeatures, uint catFeaturesSize,
            [Out] double[] result, uint resultSize
        );

        [DllImport(DynamicLibraryName)]
        public static extern bool CalcModelPredictionSingle(
            [In] IntPtr modelHandle,
            float[] floatFeatures, uint floatFeaturesSize,
            string[] catFeatures, uint catFeaturesSize,
            [Out] double[] result, uint resultSize
        );
    }
}
