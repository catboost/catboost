using System;
using System.Runtime.InteropServices;

namespace CatBoostNet
{
    /// <summary>
    /// Helper class for pointer-level memory management.
    /// </summary>
    internal class PointerTools
    {
        /// <summary>
        /// Allocate the 2D array of floats to the contiguous memory segment
        /// </summary>
        /// <param name="vals">Array to be allocated</param>
        /// <returns>Pointer to the first memory cell storing input array</returns>
        public static IntPtr AllocateToPtr(float[,] vals)
        {
            int ptrSize = Marshal.SizeOf(IntPtr.Zero);
            IntPtr ret = Marshal.AllocHGlobal(vals.GetLength(0) * ptrSize);
            int offset = 0;

            for (int i = 0; i < vals.GetLength(0); i++)
            {
                IntPtr lineRet = Marshal.AllocHGlobal(vals.GetLength(1) * sizeof(float));

                float[] arr = new float[vals.GetLength(1)];
                for (int j = 0; j < vals.GetLength(1); j++)
                {
                    arr[j] = vals[i, j];
                }
                Marshal.Copy(arr, 0, lineRet, arr.Length);
                Marshal.WriteIntPtr(ret, offset, lineRet);
                offset += ptrSize;
            }

            return ret;
        }

        /// <summary>
        /// Allocate the 2D array of strings as the contiguous array of char pointers
        /// </summary>
        /// <param name="vals">Array to be allocated</param>
        /// <returns>Pointer to the first memory cell storing array of pointers to the strings of the input array</returns>
        public static IntPtr AllocateToPtr(string[,] vals)
        {
            int ptrSize = Marshal.SizeOf(IntPtr.Zero);
            IntPtr ret = Marshal.AllocHGlobal(vals.GetLength(0) * ptrSize);
            int offset = 0;

            for (int i = 0; i < vals.GetLength(0); i++)
            {
                IntPtr lineRet = Marshal.AllocHGlobal(vals.GetLength(1) * ptrSize);

                IntPtr[] arr = new IntPtr[vals.GetLength(1)];
                for (int j = 0; j < vals.GetLength(1); j++)
                {
                    arr[j] = Marshal.StringToHGlobalUni(vals[i, j]);
                }
                Marshal.Copy(arr, 0, lineRet, arr.Length);
                Marshal.WriteIntPtr(ret, offset, lineRet);
                offset += ptrSize;
            }

            return ret;
        }
    }
}
