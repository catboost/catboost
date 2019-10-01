using System;
using System.Runtime.InteropServices;
using System.Text;

namespace CatBoostNet {
    /// <summary>
    /// Helper class for pointer-level memory management.
    /// </summary>
    internal class StringPointerHolder : IDisposable {
        private IntPtr Utf8StringBuf;
        private IntPtr StringPointerBuf;
        private IntPtr LinePointerBuf;

        public IntPtr MainPointer { get => LinePointerBuf; }

        public StringPointerHolder(string[,] vals) {
            if (vals.Length == 0) {
                this.Utf8StringBuf = IntPtr.Zero;
                this.StringPointerBuf = IntPtr.Zero;
                this.LinePointerBuf = IntPtr.Zero;
                return;
            }
            int ptrSize = Marshal.SizeOf(IntPtr.Zero);
            Encoding utf8 = Encoding.UTF8;
            int neededBufSize = 0;
            int maxStringLen = 0;
            foreach (string s in vals) {
                int currStrUtf8Size = utf8.GetByteCount(s);
                maxStringLen = Math.Max(maxStringLen, currStrUtf8Size);
                neededBufSize += currStrUtf8Size + 1;
            }
            LinePointerBuf = Marshal.AllocHGlobal(vals.GetLength(0) * ptrSize);
            StringPointerBuf = Marshal.AllocHGlobal(vals.Length * ptrSize);
            IntPtr linePointerLast = LinePointerBuf;
            IntPtr stringPointerLast = StringPointerBuf;

            this.Utf8StringBuf = Marshal.AllocHGlobal(neededBufSize);
            byte[] utf8EncodeBuf = new byte[maxStringLen + 1];
            IntPtr writePosition = Utf8StringBuf;
            for (int i = 0; i < vals.GetLength(0); ++i) {
                Marshal.WriteIntPtr(linePointerLast, stringPointerLast);
                linePointerLast += ptrSize;
                for (int j = 0; j < vals.GetLength(1); ++j) {
                    int strLen = utf8.GetBytes(vals[i, j], 0, vals[i, j].Length, utf8EncodeBuf, 0);
                    Marshal.WriteIntPtr(stringPointerLast, writePosition);
                    stringPointerLast += ptrSize;
                    Marshal.Copy(utf8EncodeBuf, 0, writePosition, strLen);
                    writePosition += strLen;
                    Marshal.WriteByte(writePosition, 0);
                    writePosition += 1;
                }
            }
        }
        public void Dispose() {
            if (Utf8StringBuf != IntPtr.Zero) {
                Marshal.FreeHGlobal(Utf8StringBuf);
                Marshal.FreeHGlobal(StringPointerBuf);
                Marshal.FreeHGlobal(LinePointerBuf);
            }
        }
    }
}
