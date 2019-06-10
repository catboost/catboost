using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace CatBoostNetTests.Schemas
{
    public class IrisDataPoint
    {
        [LoadColumn(0)]
        public float SepalLength;

        [LoadColumn(1)]
        public float SepalWidth;

        [LoadColumn(2)]
        public float PetalLength;

        [LoadColumn(3)]
        public float PetalWidth;

        [LoadColumn(4)]
        public string IrisType;
    }
}
