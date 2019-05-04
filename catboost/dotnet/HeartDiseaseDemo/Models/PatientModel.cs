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
            Uprising = 1, Flat, Downrising
        }

        public enum ThalType
        {
            Normal = 3, FixedDefect = 6, ReversableDefect = 7
        }

        public int Age { get; set; }

        public Gender Sex { get; set; }

        [DisplayName("Chest pain type")]
        public ChestPainType ChestPain { get; set; }

        [DisplayName("Resting blood pressure (mm Hg)")]
        public int BloodPressure { get; set; }

        [DisplayName("Serum cholestoral (mg/dl)")]
        public int Cholesterol { get; set; }

        [DisplayName("High blood sugar?")]
        public bool HighBloodSugar { get; set; }

        [DisplayName("ECG results")]
        public ECGResultType ECGResults { get; set; }

        [DisplayName("Max heart rate")]
        public int MaxHeartRate { get; set; }

        [DisplayName("Exerecise induced angina")]
        public bool Exang { get; set; }

        [DisplayName("ST depression")]
        public int OldPeak { get; set; }

        [DisplayName("The slope of the peak exercise ST segment")]
        public SlopeType SlopeST { get; set; }

        [DisplayName("# major vessels colored by flourosopy")]
        public int FlurMajorVessels { get; set; }

        [DisplayName("Thal type")]
        public ThalType Thal { get; set; }

        public double? Target { get; set; } = null;
        public bool IsEmpty { get; set; } = true;
    }
}
