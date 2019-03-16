using System;
using System.Runtime.InteropServices;

namespace CatBoostNet
{
    internal class PointerTools
    {
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
