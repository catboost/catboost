import os
import tempfile

import config
import test_helpers


def testPredictionValuesChange():
    spark = test_helpers.getOrCreateSparkSession(test_helpers.getCurrentMethodName())
    import catboost_spark

    dataDir = os.path.join(config.CATBOOST_TEST_DATA_DIR, 'higgs')

    trainPool = catboost_spark.Pool.load(
      spark,
      dataPathWithScheme = os.path.join(dataDir, "train_small"),
      columnDescription = os.path.join(dataDir, "train.cd")
    )

    regressor = (catboost_spark.CatBoostRegressor()
      .setIterations(20)
      .setTrainDir(tempfile.mkdtemp(prefix=test_helpers.getCurrentMethodName())))
    model = regressor.fit(trainPool)


    calcTypes = [
        catboost_spark.ECalcTypeShapValues.Regular,
        catboost_spark.ECalcTypeShapValues.Approximate,
        catboost_spark.ECalcTypeShapValues.Exact
    ]

    for calcType in calcTypes:
        featureImportancesPredictionValuesChange = model.getFeatureImportance(
            fstrType=catboost_spark.EFstrType.PredictionValuesChange,
            calcType=calcType
        )
        print('calcType=' + str(calcType) + ',featureImportancesPredictionValuesChange=')
        print(featureImportancesPredictionValuesChange)

        featureImportancesDefault = model.getFeatureImportance(calcType=calcType)
        print('calcType=' + str(calcType) + ',featureImportancesDefault=')
        print(featureImportancesDefault)

        featureImportancesPredictionValuesChangePrettified = model.getFeatureImportancePrettified(
            fstrType=catboost_spark.EFstrType.PredictionValuesChange,
            calcType=calcType
        )
        print('calcType=' + str(calcType) + ',featureImportancesPredictionValuesChangePrettified=')
        for e in featureImportancesPredictionValuesChangePrettified:
            print ('featureName={},importance={}'.format(e.featureName(), e.importance()))

        featureImportancesDefaultPrettified = model.getFeatureImportancePrettified(calcType=calcType)
        print('calcType=' + str(calcType) + ',featureImportancesDefaultPrettified=')
        for e in featureImportancesDefaultPrettified:
            print ('featureName={},importance={}'.format(e.featureName(), e.importance()))


def testLossFunctionChange():
    spark = test_helpers.getOrCreateSparkSession(test_helpers.getCurrentMethodName())
    import catboost_spark

    dataDir = os.path.join(config.CATBOOST_TEST_DATA_DIR, 'querywise')

    trainPool = catboost_spark.Pool.load(
      spark,
      dataPathWithScheme = os.path.join(dataDir, "train"),
      columnDescription = os.path.join(dataDir, "train.cd")
    )

    regressor = (catboost_spark.CatBoostRegressor()
      .setIterations(20)
      .setLossFunction("QueryRMSE")
      .setTrainDir(tempfile.mkdtemp(prefix=test_helpers.getCurrentMethodName())))
    model = regressor.fit(trainPool)

    calcTypes = [
        catboost_spark.ECalcTypeShapValues.Regular,
        catboost_spark.ECalcTypeShapValues.Approximate,
        catboost_spark.ECalcTypeShapValues.Exact
    ]

    for calcType in calcTypes:
        featureImportancesPredictionValuesChange = model.getFeatureImportance(
            fstrType=catboost_spark.EFstrType.LossFunctionChange,
            data=trainPool,
            calcType=calcType
        )
        print('calcType=' + str(calcType) + ',featureImportancesLossFunctionChange=')
        print(featureImportancesPredictionValuesChange)

        featureImportancesDefault = model.getFeatureImportance(data=trainPool, calcType=calcType)
        print('calcType=' + str(calcType) + ',featureImportancesDefault=')
        print(featureImportancesDefault)

        featureImportancesPredictionValuesChangePrettified = model.getFeatureImportancePrettified(
            fstrType=catboost_spark.EFstrType.LossFunctionChange,
            data=trainPool,
            calcType=calcType
        )
        print('calcType=' + str(calcType) + ',featureImportancesLossFunctionChangePrettified=')
        for e in featureImportancesPredictionValuesChangePrettified:
            print ('featureName={},importance={}'.format(e.featureName(), e.importance()))

        featureImportancesDefaultPrettified = model.getFeatureImportancePrettified(
            data=trainPool,
            calcType=calcType
        )
        print('calcType=' + str(calcType) + ',featureImportancesDefaultPrettified=')
        for e in featureImportancesDefaultPrettified:
            print ('featureName={},importance={}'.format(e.featureName(), e.importance()))

def testInteraction():
    spark = test_helpers.getOrCreateSparkSession(test_helpers.getCurrentMethodName())
    import catboost_spark

    dataDir = os.path.join(config.CATBOOST_TEST_DATA_DIR, 'querywise')

    trainPool = catboost_spark.Pool.load(
      spark,
      dataPathWithScheme = os.path.join(dataDir, "train"),
      columnDescription = os.path.join(dataDir, "train.cd")
    )

    regressor = (catboost_spark.CatBoostRegressor()
      .setIterations(20)
      .setLossFunction("QueryRMSE")
      .setTrainDir(tempfile.mkdtemp(prefix=test_helpers.getCurrentMethodName())))
    model = regressor.fit(trainPool)

    featureImportancesInteraction = model.getFeatureImportanceInteraction()
    for e in featureImportancesInteraction:
        print(
            'firstFeatureIdx={},secondFeatureIdx={},score={}'.format(
                e.firstFeatureIdx(),
                e.secondFeatureIdx(),
                e.score()
            )
        )


def shapValuesTestCase(problemType, model, data):
    import catboost_spark
    shapModes = [
        catboost_spark.EPreCalcShapValues.Auto,
        catboost_spark.EPreCalcShapValues.UsePreCalc,
        catboost_spark.EPreCalcShapValues.NoPreCalc
    ]
    calcTypes = [
        catboost_spark.ECalcTypeShapValues.Regular,
        catboost_spark.ECalcTypeShapValues.Approximate,
        catboost_spark.ECalcTypeShapValues.Exact
    ]

    for shapMode in shapModes:
        for calcType in calcTypes:
            print ("problem_type={},shap_mode={},shap_calc_type={}".format(problemType, shapMode, calcType))
            shapValuesDf = model.getFeatureImportanceShapValues(
                data=data,
                preCalcMode=shapMode,
                calcType=calcType
            )
            shapValuesDf.show(truncate=False)

def testShapValuesForBinClass():
    spark = test_helpers.getOrCreateSparkSession(test_helpers.getCurrentMethodName())
    import catboost_spark

    dataDir = os.path.join(config.CATBOOST_TEST_DATA_DIR, 'higgs')

    trainPool = catboost_spark.Pool.load(
      spark,
      dataPathWithScheme = os.path.join(dataDir, "train_small"),
      columnDescription = os.path.join(dataDir, "train.cd")
    )

    classifier = (catboost_spark.CatBoostClassifier()
      .setIterations(20)
      .setLossFunction("Logloss")
      .setTrainDir(tempfile.mkdtemp(prefix=test_helpers.getCurrentMethodName())))
    model = classifier.fit(trainPool)

    shapValuesTestCase('BinClass', model, trainPool)

def testShapValuesForMultiClass():
    spark = test_helpers.getOrCreateSparkSession(test_helpers.getCurrentMethodName())
    import catboost_spark

    dataDir = os.path.join(config.CATBOOST_TEST_DATA_DIR, 'cloudness_small')

    trainPool = catboost_spark.Pool.load(
      spark,
      dataPathWithScheme = os.path.join(dataDir, "train_small"),
      columnDescription = os.path.join(dataDir, "train_float.cd")
    )

    classifier = (catboost_spark.CatBoostClassifier()
      .setIterations(20)
      .setLossFunction("MultiClass")
      .setTrainDir(tempfile.mkdtemp(prefix=test_helpers.getCurrentMethodName())))
    model = classifier.fit(trainPool)

    shapValuesTestCase('MultiClass', model, trainPool)

def testShapValuesForRegression():
    spark = test_helpers.getOrCreateSparkSession(test_helpers.getCurrentMethodName())
    import catboost_spark

    dataDir = os.path.join(config.CATBOOST_TEST_DATA_DIR, 'querywise')

    trainPool = catboost_spark.Pool.load(
      spark,
      dataPathWithScheme = os.path.join(dataDir, "train"),
      columnDescription = os.path.join(dataDir, "train.cd")
    )

    regressor = (catboost_spark.CatBoostRegressor()
      .setIterations(20)
      .setLossFunction("QueryRMSE")
      .setTrainDir(tempfile.mkdtemp(prefix=test_helpers.getCurrentMethodName())))
    model = regressor.fit(trainPool)

    shapValuesTestCase('Regression', model, trainPool)


def testPredictionDiff():
    spark = test_helpers.getOrCreateSparkSession(test_helpers.getCurrentMethodName())
    import catboost_spark

    dataDir = os.path.join(config.CATBOOST_TEST_DATA_DIR, 'higgs')

    trainPool = catboost_spark.Pool.load(
      spark,
      dataPathWithScheme = os.path.join(dataDir, "train_small"),
      columnDescription = os.path.join(dataDir, "train.cd")
    )
    dataForPredictionDiff = catboost_spark.Pool(trainPool.data.limit(2))

    regressor = (catboost_spark.CatBoostRegressor()
      .setIterations(20)
      .setTrainDir(tempfile.mkdtemp(prefix=test_helpers.getCurrentMethodName())))
    model = regressor.fit(trainPool)


    featureImportances = model.getFeatureImportance(
        fstrType=catboost_spark.EFstrType.PredictionDiff,
        data=dataForPredictionDiff
    )
    print('featureImportancesPredictionDiff=')
    print(featureImportances)

    featureImportancesPrettified = model.getFeatureImportancePrettified(
        fstrType=catboost_spark.EFstrType.PredictionDiff,
        data=dataForPredictionDiff
    )

    print('featureImportancesPredictionDiffPrettified=')
    for e in featureImportancesPrettified:
        print('featureName={},importance={}'.format(e.featureName(), e.importance()))


def shapInteractionValuesTestCase(problemType, model, data):
    import catboost_spark

    dataForFeatureImportance = catboost_spark.Pool(data.data.limit(5))

    shapModes = [
        catboost_spark.EPreCalcShapValues.Auto,
        catboost_spark.EPreCalcShapValues.UsePreCalc,
        catboost_spark.EPreCalcShapValues.NoPreCalc
    ]
    calcTypes = [
        catboost_spark.ECalcTypeShapValues.Regular
    ]

    for shapMode in shapModes:
        for calcType in calcTypes:
            print ("problem_type={},shap_mode={},shap_calc_type={}".format(problemType, shapMode, calcType))
            shapInteractionValuesDf = model.getFeatureImportanceShapInteractionValues(
                data=dataForFeatureImportance,
                preCalcMode=shapMode,
                calcType=calcType
            )
            shapInteractionValuesDf.show(truncate=False)

def testShapInteractionValuesForBinClass():
    spark = test_helpers.getOrCreateSparkSession(test_helpers.getCurrentMethodName())
    import catboost_spark

    dataDir = os.path.join(config.CATBOOST_TEST_DATA_DIR, 'higgs')

    trainPool = catboost_spark.Pool.load(
      spark,
      dataPathWithScheme = os.path.join(dataDir, "train_small"),
      columnDescription = os.path.join(dataDir, "train.cd")
    )

    classifier = (catboost_spark.CatBoostClassifier()
      .setIterations(20)
      .setLossFunction("Logloss")
      .setTrainDir(tempfile.mkdtemp(prefix=test_helpers.getCurrentMethodName())))
    model = classifier.fit(trainPool)

    shapInteractionValuesTestCase('BinClass', model, trainPool)

def testShapInteractionValuesForMultiClass():
    spark = test_helpers.getOrCreateSparkSession(test_helpers.getCurrentMethodName())
    import catboost_spark

    dataDir = os.path.join(config.CATBOOST_TEST_DATA_DIR, 'cloudness_small')

    trainPool = catboost_spark.Pool.load(
      spark,
      dataPathWithScheme = os.path.join(dataDir, "train_small"),
      columnDescription = os.path.join(dataDir, "train_float.cd")
    )

    classifier = (catboost_spark.CatBoostClassifier()
      .setIterations(20)
      .setLossFunction("MultiClass")
      .setTrainDir(tempfile.mkdtemp(prefix=test_helpers.getCurrentMethodName())))
    model = classifier.fit(trainPool)

    shapInteractionValuesTestCase('MultiClass', model, trainPool)

def testShapInteractionValuesForRegression():
    spark = test_helpers.getOrCreateSparkSession(test_helpers.getCurrentMethodName())
    import catboost_spark

    dataDir = os.path.join(config.CATBOOST_TEST_DATA_DIR, 'higgs')

    trainPool = catboost_spark.Pool.load(
      spark,
      dataPathWithScheme = os.path.join(dataDir, "train_small"),
      columnDescription = os.path.join(dataDir, "train.cd")
    )

    regressor = (catboost_spark.CatBoostRegressor()
      .setIterations(20)
      .setLossFunction("RMSE")
      .setTrainDir(tempfile.mkdtemp(prefix=test_helpers.getCurrentMethodName())))
    model = regressor.fit(trainPool)

    shapInteractionValuesTestCase('Regression', model, trainPool)
