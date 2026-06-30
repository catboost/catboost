using System;
using System.Runtime.InteropServices;

namespace CatBoostNet {
    /// <summary>
    /// Static class containing definitions for imported functions
    /// </summary>
    internal static class CatboostNativeInterface {
#if _WINDOWS
        public const string DynamicLibraryName = "catboostmodel";
#else
        public const string DynamicLibraryName = "libcatboostmodel";
#endif

        /// <summary>
        /// Create native model handle
        /// </summary>
        /// <returns>Handle (pointer) to empty model</returns>
        [DllImport(DynamicLibraryName)]
        public static extern IntPtr ModelCalcerCreate();

        [DllImport(DynamicLibraryName)]
        private static extern IntPtr GetErrorString([In] IntPtr handler);

        /// <summary>
        /// Get error string from model library
        /// </summary>
        /// <param name="handler"></param>
        /// <returns>String with textual error description</returns>
        public static string GetErrorStringConst([In] IntPtr handler) {
            var ptr = GetErrorString(handler);
            return PtrToStringUtf8(ptr);
        }

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

        [DllImport(DynamicLibraryName)]
        private static extern IntPtr GetModelInfoValue([In] IntPtr handler, string key, uint keySize);

        /// <summary>
        /// Get model metainfo for some key
        /// </summary>
        /// <param name="handler"></param>
        /// <param name="key"></param>
        /// <param name="keySize"></param>
        /// <returns>String with metainfo description</returns>
        public static string GetModelInfoValueConst([In] IntPtr handler, string key, uint keySize) {
            var ptr = GetModelInfoValue(handler, key, keySize);
            return PtrToStringUtf8(ptr);
        }

        /// <summary>
        /// Use CUDA GPU device for model evaluation
        /// </summary>
        /// <param name="handler"></param>
        /// <param name="deviceId"></param>
        /// <returns>False if error occured</returns>
        [DllImport(DynamicLibraryName)]
        public static extern bool EnableGPUEvaluation([In] IntPtr handler, int deviceId);

        private static string PtrToStringUtf8(IntPtr ptr)
        {
            if (ptr == IntPtr.Zero)
                return "";
            int len = 0;
            while (Marshal.ReadByte(ptr, len) != 0)  // string message is zero-terminated
                len++;
            if (len == 0)
                return "";
            var array = new byte[len];
            Marshal.Copy(ptr, array, 0, len);
            return System.Text.Encoding.UTF8.GetString(array);
        }

    }
}
