using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace CatBoostNetTests.Schemas
{
    public class BostonDataPoint
    {
        [LoadColumn(0)]
        public float Crim;

        [LoadColumn(1)]
        public float Zn;

        [LoadColumn(2)]
        public float Indus;

        [LoadColumn(3)]
        public float Chas;

        [LoadColumn(4)]
        public float Nox;

        [LoadColumn(5)]
        public float Rm;

        [LoadColumn(6)]
        public float Age;

        [LoadColumn(7)]
        public float Dis;

        [LoadColumn(8)]
        public float Rad;

        [LoadColumn(9)]
        public float Tax;

        [LoadColumn(10)]
        public float PTRatio;

        [LoadColumn(11)]
        public float B;

        [LoadColumn(12)]
        public float LStat;

        [LoadColumn(13)]
        public float MedV;
    }
}
