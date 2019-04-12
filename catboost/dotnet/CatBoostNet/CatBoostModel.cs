using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Linq;

namespace CatBoostNet
{
    public class CatBoostValuePrediction
    {
        [ColumnName("Prediction")]
        public double[] OutputValues;
    }

    public class CatBoostModel : ITransformer
    {
        private CatBoostModelEvaluator Evaluator { get; }
        public string TargetFeature { get; }
        public MLContext Context { get; }

        public CatBoostModel(string modelFilePath, string targetFeature, MLContext context)
        {
            Evaluator = new CatBoostModelEvaluator(modelFilePath);
            TargetFeature = targetFeature;
            Context = context;
        }

        public bool IsRowToRowMapper => true;

        public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
        {
            throw new NotImplementedException();
        }

        public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        {
            throw new NotImplementedException();
        }

        public void Save(ModelSaveContext ctx)
        {
            // Nothing to do here, we are loading model from the file anyway ;)
        }

        public IDataView Transform(IDataView input)
        {
            int nFloat = 0, nCat = 0;
            Collection<int> catFeatureIxs = new Collection<int>();

            using (var cursor = input.GetRowCursor(input.Schema))
            {
                cursor.MoveNext();
                int ix = -1;
                int ptr = 0;
                foreach (var col in cursor.Schema)
                {
                    ix++;
                    if (col.Name == TargetFeature)
                    {
                        continue;
                    }

                    string typeId = col.Type.ToString();
                    if ("IUR".Contains(typeId[0]))
                    {
                        float target = 0.0f;
                        cursor.GetGetter<float>(ix)(ref target);
                        nFloat++;
                    }
                    else if (typeId == "Text")
                    {
                        ReadOnlyMemory<char> target = new ReadOnlyMemory<char>();
                        cursor.GetGetter<ReadOnlyMemory<char>>(ix)(ref target);
                        catFeatureIxs.Add(ptr);
                        nCat++;
                    }
                    else
                    {
                        throw new NotSupportedException($"Datatype {typeId} is not supported.");
                    }
                    ptr++;
                }
            }

            List<float[]> floatFeatures = new List<float[]>();
            List<string[]> catFeatures = new List<string[]>();

            using (var cursor = input.GetRowCursor(input.Schema))
            {
                int rowIx = 0;
                while (cursor.MoveNext())
                {
                    int ix = -1;
                    int ptr = 0;
                    int realPtr = 0;
                    int catPtr = 0;

                    float[] floats = new float[nFloat];
                    string[] cats = new string[nCat];

                    foreach (var col in cursor.Schema)
                    {
                        ix++;
                        if (col.Name == TargetFeature)
                        {
                            continue;
                        }

                        string typeId = col.Type.ToString();
                        if ("IUR".Contains(typeId[0]))
                        {
                            cursor.GetGetter<float>(ix)(ref floats[realPtr]);
                            realPtr++;
                        }
                        else if (typeId == "Text")
                        {
                            Debug.Assert(catFeatureIxs.Contains(ptr));
                            ReadOnlyMemory<char> target = new ReadOnlyMemory<char>();
                            cursor.GetGetter<ReadOnlyMemory<char>>(ix)(ref target);
                            cats[catPtr] = new string(target.ToArray());
                            catPtr++;
                        }
                        else
                        {
                            throw new NotSupportedException($"Datatype {typeId} is not supported.");
                        }
                        ptr++;
                    }

                    Debug.Assert(realPtr == nFloat);
                    Debug.Assert(catPtr == nCat);

                    floatFeatures.Add(floats);
                    catFeatures.Add(cats);

                    rowIx++;
                }
            }

            float[,] floatFeaturesArray = new float[floatFeatures.Count, nFloat];
            string[,] catFeaturesArray = new string[catFeatures.Count, nCat];

            for (int i = 0; i < floatFeatures.Count; ++i)
            {
                for (int j = 0; j < nFloat; ++j)
                {
                    floatFeaturesArray[i, j] = floatFeatures[i][j];
                }
                for (int j = 0; j < nCat; ++j)
                {
                    catFeaturesArray[i, j] = catFeatures[i][j];
                }
            }

            Evaluator.CatFeaturesIndices = catFeatureIxs;
            double[,] res = Evaluator.EvaluateBatch(floatFeaturesArray, catFeaturesArray);

            return Context.Data.LoadFromEnumerable<CatBoostValuePrediction>(
                Enumerable.Range(0, res.GetLength(0)).Select(i => new CatBoostValuePrediction
                {
                    OutputValues = Enumerable.Range(0, res.GetLength(1)).Select(j => res[i, j]).ToArray()
                })
            );
        }
    }
}
