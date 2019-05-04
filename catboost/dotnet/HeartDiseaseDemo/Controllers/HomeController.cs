using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using HeartDiseaseDemo.Models;
using Microsoft.AspNetCore.Mvc.ModelBinding;

namespace HeartDiseaseDemo.Controllers
{
    public class HomeController : Controller
    {
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
                if (!ModelState.IsValid)
                {
                    ViewBag.Output = null;
                }
                else
                {
                    ViewBag.Output = 2.15d;
                }
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
