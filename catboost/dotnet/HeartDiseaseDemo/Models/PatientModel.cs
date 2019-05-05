using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Threading.Tasks;

namespace HeartDiseaseDemo.Models
{
    public class PatientModel
    {
        public enum Gender
        {
            Male = 1,
            Female = 0
        }

        public enum ChestPainType
        {
            Type1, Type2, Type3, Type4
        }

        public enum ECGResultType
        {
            Normal, STTWaveAbnormality, LeftVentricularHypertrophy
        }

        public enum SlopeType
        {
            Uprising, Flat, Downrising
        }

        public enum ThalType
        {
            Normal = 1, FixedDefect = 2, ReversibleDefect = 3
        }

        [LoadColumn(0)]
        public int Age { get; set; }

        [NoColumn]
        public Gender Sex { get; set; }

        [LoadColumn(1)]
        public string SexCat => ((int)Sex).ToString();

        [DisplayName("Chest pain type")]
        [NoColumn]
        public ChestPainType ChestPain { get; set; }

        [LoadColumn(2)]
        public string ChestPainCat => ((int)ChestPain).ToString();

        [LoadColumn(3)]
        [DisplayName("Resting blood pressure (mm Hg)")]
        public int BloodPressure { get; set; }

        [LoadColumn(4)]
        [DisplayName("Serum cholestoral (mg/dl)")]
        public int Cholesterol { get; set; }

        [LoadColumn(5)]
        [DisplayName("High blood sugar?")]
        public bool HighBloodSugar { get; set; }

        [DisplayName("ECG results")]
        [NoColumn]
        public ECGResultType ECGResults { get; set; }

        [LoadColumn(6)]
        public string ECGResultsCat => ((int)ECGResults).ToString();

        [LoadColumn(7)]
        [DisplayName("Max heart rate")]
        public int MaxHeartRate { get; set; }

        [LoadColumn(8)]
        [DisplayName("Exercise induced angina")]
        public bool Exang { get; set; }

        [LoadColumn(9)]
        [DisplayName("ST depression")]
        public float OldPeak { get; set; }

        [DisplayName("The slope of the peak exercise ST segment")]
        [NoColumn]
        public SlopeType SlopeST { get; set; }

        [LoadColumn(10)]
        public string SlopeSTCat => ((int)SlopeST).ToString();

        [LoadColumn(11)]
        [DisplayName("# major vessels colored by flourosopy")]
        public int FlurMajorVessels { get; set; }

        [DisplayName("Thal type")]
        [NoColumn]
        public ThalType Thal { get; set; }

        [LoadColumn(12)]
        public string ThalCat => ((int)Thal).ToString();

        public double Target { get; set; }
        public bool IsEmpty { get; set; } = true;
    }
}
