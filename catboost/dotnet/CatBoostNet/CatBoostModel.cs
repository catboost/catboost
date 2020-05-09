using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Linq;

namespace CatBoostNet
{
    /// <summary>
    /// Schema for the CatBoost model prediction.
    /// Currently it stores the array of doubles which can mean:
    /// <list type="bullet">
    /// <item>response variable (1 real number) for regression;</item>
    /// <item>logit of probability estimate (1 real number) for binary classification;</item>
    /// <item>logits of probability estimates (several real numbers, one for each class) for multiclass classification</item>
    /// </list>
    /// </summary>
    public class CatBoostValuePrediction
    {
        /// <summary>
        /// Array of output values
        /// </summary>
        [ColumnName("Prediction")]
        public double[] OutputValues;
    }

    /// <summary>
    /// High-level class for making predictions using CatBoost models.
    /// If you need to implement a custom model code, use <see cref="CatBoostModelEvaluator"/>
    /// as the basic building block.
    /// </summary>
    public class CatBoostModel : IDisposable
    {
        private CatBoostModelEvaluator Evaluator { get; }

        /// <summary>
        /// Name of the predicted feature
        /// </summary>
        public string TargetFeature { get; }

        /// <summary>
        /// Model constructor
        /// </summary>
        /// <param name="modelFilePath">Path to the model file</param>
        /// <param name="targetFeature">Name of the target feature</param>
        public CatBoostModel(string modelFilePath, string targetFeature)
        {
            Evaluator = new CatBoostModelEvaluator(modelFilePath);
            TargetFeature = targetFeature;
        }

        /// <inheritdoc />
        public IDataView Transform(IDataView input)
        {
            HashSet<string> numTypes = new HashSet<string>
            {
                "Boolean", "Single", "Int32"
            };

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
                    if (numTypes.Contains(typeId))
                    {
                        nFloat++;
                    }
                    else if (typeId == "String")
                    {
                        catFeatureIxs.Add(ptr);
                        nCat++;
                    }
                    else
                    {
                        throw new NotSupportedException($"Data type {typeId} is not supported.");
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
                        if (numTypes.Contains(typeId))
                        {
                            object value = new object();

                            switch (typeId)
                            {
                                case "Boolean":
                                {
                                    bool receivedVal = new bool();
                                    cursor.GetGetter<bool>(col)(ref receivedVal);
                                    value = receivedVal;
                                    break;
                                }
                                case "Single":
                                {
                                    float receivedVal = new float();
                                    cursor.GetGetter<float>(col)(ref receivedVal);
                                    value = receivedVal;
                                    break;
                                }
                                case "Int32":
                                {
                                    int receivedVal = new int();
                                    cursor.GetGetter<int>(col)(ref receivedVal);
                                    value = receivedVal;
                                    break;
                                }
                            }

                            floats[realPtr] = Convert.ToSingle(value);
                            realPtr++;
                        }
                        else if (typeId == "String")
                        {
                            Debug.Assert(catFeatureIxs.Contains(ptr));
                            ReadOnlyMemory<char> target = new ReadOnlyMemory<char>();
                            cursor.GetGetter<ReadOnlyMemory<char>>(col)(ref target);
                            cats[catPtr] = new string(target.ToArray());
                            catPtr++;
                        }
                        else
                        {
                            throw new NotSupportedException($"Data type {typeId} is not supported.");
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

            return (new MLContext()).Data.LoadFromEnumerable(
                Enumerable.Range(0, res.GetLength(0)).Select(i => new CatBoostValuePrediction
                {
                    OutputValues = Enumerable.Range(0, res.GetLength(1)).Select(j => res[i, j]).ToArray()
                })
            );
        }

        /// <summary>
        /// Dispose of unmanaged resources
        /// </summary>
        public void Dispose()
        {
            Evaluator?.Dispose();
        }
    }
}
