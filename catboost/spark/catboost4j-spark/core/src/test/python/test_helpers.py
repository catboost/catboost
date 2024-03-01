import inspect

import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.types import *


def getCurrentMethodName():
    return inspect.currentframe().f_back.f_code.co_name


def getOrCreateSparkSession(appName):
    return (SparkSession.builder
        .master("local[2]")
        .config("spark.jars.packages", "ai.catboost:catboost-spark_2.4_2.11:1.2.3")
        .appName(appName)
        .getOrCreate()
    )

def getDataForComparison(data, sortByFields):
    if not sortByFields:
        return data
    else:
        return data.sort(sortByFields).collect()


def assertEqualsWithPrecision(expected, actual, sortByFields=[]):
    assert expected.count() == actual.count()

    print("expected.schema")
    expected.schema().printTreeString()

    print("actual.schema")
    actual.schema().printTreeString()

    assert expected.schema().size() == actual.schema().size()
    for i in range(expected.schema().size()):
        expectedField = expected.schema()[i]
        actualField = actual.schema()[i]
        assert expectedField.name == actualField.name
        assert expectedField.dataType == actualField.dataType
        assert expectedField.nullable == actualField.nullable
        assert expectedField.metadata == actualField.metadata


    schema = expected.schema()
    rowSize = schema.size()

    expectedRows = getDataForComparison(expected, sortByFields)
    actualRows = getDataForComparison(actual, sortByFields)

    for rowIdx in range(expectedRows.size()):
      expectedRow = expectedRows[rowIdx]
      actualRow = actualRows[rowIdx]

      assert rowSize == expectedRow.size()
      assert rowSize == actualRow.size()

      for fieldIdx in range(rowSize):
          if schema[fieldIdx].dataType == FloatType:
              assert abs(expectedRow[fieldIdx] - actualRow[fieldIdx]) < 1e-5
          elif schema[fieldIdx].dataType == DoubleType:
              assert abs(expectedRow[fieldIdx] - actualRow[fieldIdx]) < 1e-6
          elif schema[fieldIdx].dataType == SQLDataTypes.VectorType:
              assert np.isclose(expectedRow[fieldIdx].toArray(), actualRow[fieldIdx].toArray())
          else:
              assert expectedRow[fieldIdx] == actualRow[fieldIdx]
