using System;
using System.Collections.Generic;
using System.Net;
using System.Text;

namespace CatBoostNetTests
{
    public class DownloadHelpers
    {
        public static void DownloadDataset(string url, string outFile)
        {
            using (var client = new WebClient())
            {
                client.DownloadFile(url, outFile);
            }
        }
    }
}
