using CatBoostNet;
using HeartDiseaseDemo.Helpers;
using HeartDiseaseDemo.Models;
using Microsoft.AspNetCore.Mvc;
using System;
using System.Diagnostics;

namespace HeartDiseaseDemo.Controllers
{
    public class HomeController : Controller
    {
        public CatBoostModel Predictor { get; } = new CatBoostModel("predictor.cbm", "Target");

        public IActionResult Index([FromQuery]PatientModel model)
        {
            ViewBag.Sex = model.IsEmpty ? PatientModel.Gender.Male : model.Sex;
            ViewBag.ChestPain = model.IsEmpty ? PatientModel.ChestPainType.Type1 : model.ChestPain;
            ViewBag.ECGResults = model.IsEmpty ? PatientModel.ECGResultType.STTWaveAbnormality : model.ECGResults;
            ViewBag.SlopeST = model.IsEmpty ? PatientModel.SlopeType.Uprising : model.SlopeST;
            ViewBag.Thal = model.IsEmpty ? PatientModel.ThalType.Normal : model.Thal;
            ViewBag.FlurMajorVessels = model.IsEmpty ? 0 : model.FlurMajorVessels;

            ViewBag.Exang = model.IsEmpty ? false : model.Exang;
            ViewBag.HighBloodSugar = model.IsEmpty ? false : model.HighBloodSugar;

            if (!model.IsEmpty)
            {
                // https://localhost:44396/?IsEmpty=False&Age=63&Sex=Male&ChestPain=Type3&BloodPressure=145&Cholesterol=233&HighBloodSugar=true&ECGResults=Normal&MaxHeartRate=150&OldPeak=2.3&SlopeST=Uprising&FlurMajorVessels=0&Thal=Normal&HighBloodSugar=true&Exang=false
                ViewBag.Output = ModelState.IsValid
                    ? new Nullable<double>(PredictionHelpers.Predict(model, Predictor))
                    : null;
            }

            return View();
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
    }
}
