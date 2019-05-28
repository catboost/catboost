using System;
using System.Collections.Generic;
using System.Text;

namespace CatBoostNet
{
    class CatBoostException : Exception {
        public CatBoostException()
        {
        }

        public CatBoostException(string message)
            : base(message)
        {
        }

        public CatBoostException(string message, Exception inner)
            : base(message, inner)
        {
        }
    }
}
