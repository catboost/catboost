using Microsoft.ML.Data;

namespace CatBoostNetTests.Schemas
{
    public class MushroomDataPoint
    {
        [LoadColumn(0)]
        public string Class;

        [LoadColumn(1)]
        public string CapShape;

        [LoadColumn(2)]
        public string CapSurface;

        [LoadColumn(3)]
        public string CapColor;

        [LoadColumn(4)]
        public string Bruises;

        [LoadColumn(5)]
        public string Odor;

        [LoadColumn(6)]
        public string GillAttachment;

        [LoadColumn(7)]
        public string GillSpacing;

        [LoadColumn(8)]
        public string GillSize;

        [LoadColumn(9)]
        public string GillColor;

        [LoadColumn(10)]
        public string StalkShape;

        [LoadColumn(11)]
        public string StalkRoot;

        [LoadColumn(12)]
        public string StalkSurfaceAboveRing;

        [LoadColumn(13)]
        public string StalkSurfaceBelowRing;

        [LoadColumn(14)]
        public string StalkColorAboveRing;

        [LoadColumn(15)]
        public string StalkColorBelowRing;

        [LoadColumn(16)]
        public string VeilType;

        [LoadColumn(17)]
        public string VeilColor;

        [LoadColumn(18)]
        public string RingNumber;

        [LoadColumn(19)]
        public string RingType;

        [LoadColumn(20)]
        public string SporePrintColor;

        [LoadColumn(21)]
        public string Population;

        [LoadColumn(22)]
        public string Habitat;
    }
}
