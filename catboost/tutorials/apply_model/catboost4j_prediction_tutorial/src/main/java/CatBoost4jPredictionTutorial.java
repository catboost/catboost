import ai.catboost.CatBoostError;
import ai.catboost.CatBoostModel;
import ai.catboost.CatBoostPredictions;

import java.io.IOException;

public class CatBoost4jPredictionTutorial {
    private CatBoostModel adultModel = null;

    public CatBoost4jPredictionTutorial() throws CatBoostError, IOException {
        // Load "adult.cbm" model that we trained withing Jupyter Notebook
        adultModel = CatBoostModel.loadModel(ClassLoader.getSystemResourceAsStream("models/adult.cbm"));

        // You can also try to load your own model just comment out the line above and uncomment two lines below while
        // replacing "foo/bar" with path to your model that classifies data from UCI Adult Dataset.
        //
        // final String adultModelPath = "foo/bar";
        // adultModel = CatBoostModel.loadModel(adultModelPath);
    }

    public static double sigmoid(final double x) {
        return 1. / (1 + Math.pow(Math.E, -x));
    }

    public static String answer(final boolean makesOver50KAYear) {
        if (makesOver50KAYear) {
            return "makes over 50K a year";
        }

        return "doesn't make over 50K a year";
    }

    public void playWithModelForAdultDataset() throws CatBoostError {
        // First lets print available model metainformation.

        System.out.print("Adult dataset model metainformation\n");

        System.out.print(String.format("tree count: %d\n", adultModel.getTreeCount()));

        // In our case we were solving a binary classification problem (weather person makes over 50K a year), so the
        // dimension of the prediction will be 1, it will return probability of the object to belong to the positive
        // class; in our case we had two classed encoded as "<=50K" and ">50K", during data preprocessing (see
        // `get_fixed_adult()` in Notebook) we encoded "<=50K" as 0 and ">50K" as 1, so that ">50K" became a positive
        // class. Probability of the negative class ("<=50K") can be easily deduced as (1-p) where p is a probability of
        // positive class.
        //
        // For most of cases prediction dimension will be 1 (for regression and for ranking), it can be N for cases of
        // multiclassification, where N is a number of classes.
        System.out.print(String.format("prediction dimension: %d\n",adultModel.getPredictionDimension()));

        // Take a note, number of numeric features used by the model may be less than number of numeric features
        // that were present in a training dataset. This may happen if, for example, when traing dataset contained
        // constant features, they do not carry any information for classifier, so training process will ignore them.
        System.out.print(String.format("used numeric feature count: %d\n", adultModel.getUsedNumericFeatureCount()));

        // Number of categoric features used by the classifier may also be less than number of categoric feature present
        // in training dataset, for the same reasons as for numeric features.
        System.out.print(String.format("used categoric feature count: %d\n", adultModel.getUsedCategoricFeatureCount()));

        // Ok now lets try to use our model for prediction. We'll look at the test part of Adult dataset. You will need
        // to download it [1] from UCI repository. Look for "adult.test", "adult.name" will also be useful because it
        // in contains human-readable description of the dataset.
        //
        // So the first line of test part of the dataset is:
        //
        // "25, Private, 226802, 11th, 7, Never-married, Machine-op-inspct, Own-child, Black, Male, 0, 0, 40, United-States, <=50K."
        //
        // Based on "adult.name" we can recover its vectors of numeric and categoric features (in our case all
        // "continuous" features are numeric and all other features are categoric):
        //
        // numericFeatures: {25, 226802, 7, 0, 0, 40}
        // categoricFeatures: {"Private", "11th", "Never-married", "Machine-op-inspct", "Own-child", "Black", "Male", "United-States"}
        //
        // And he doesn't make 50K per year. Also note that order of numeric and categoric features in source data and
        // in `numericFeatures` and `categoricFeatures` is kept the same. Otherwise we can't apply the model (well, we
        // can, but result of prediction will be garbage).
        //
        // Now lets run it! And let's call this person "person A", to make variable names unique.
        //
        // [1]: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/

        System.out.print("\n");

        final float[] personANumericFeatures = new float[]{25, 226802, 7, 0, 0, 40};
        final String[] personACategoricFeatures = new String[]{"Private", "11th", "Never-married", "Machine-op-inspct", "Own-child", "Black", "Male", "United-States"};
        final CatBoostPredictions personAPrediction = adultModel.predict(personANumericFeatures, personACategoricFeatures);

        // Since we made prediction only for one person, "person A" will have index 0 in `personAPredictions`, and
        // since prediction dimension is 1. Proability of person A make over 50K is also at index 0.
        //
        // CatBoost doesn't compute "probability", to turn CatBoost prediction into a probability we'll need to apply
        // sigmoid function.
        final double personAMakesOver50KProbability = sigmoid(personAPrediction.get(0, 0));
        System.out.print(String.format("Person A make over 50K a year with probability %f\n", personAMakesOver50KProbability));

        // When we were training CatBoost we used a default classification threshold for AUC which is equal to 0.5,
        // this means that our formula is optimized for this threashold, though we may change threshold to optimize some
        // other metric on a different dataset, but we won't do it in this tutorial.
        final double classificationThreshold = 0.5;

        final boolean personAMakesOver50K = personAMakesOver50KProbability > classificationThreshold;
        System.out.print(String.format("Person A %s\n", answer(personAMakesOver50K)));

        // Now lets find an example with missing features and income greater than 50K a year. At line 40 of "adult.test"
        // we can find following line:
        //
        // "40, Private, 85019, Doctorate, 16, Married-civ-spouse, Prof-specialty, Husband, Asian-Pac-Islander, Male, 0, 0, 45, ?, >50K."
        //
        // Lets call this person "Person B", dataset missing (missing features are marked with "?") "native-county"
        // feature for Person B. When we were doing preprocessing in `get_fixed_adult` we replaced missing categoric
        // features with string "nan", now, when we apply trained model we must also use "nan" for missing features.
        // Lets write out feature vectors for Person B:
        //
        // numericFeatures = {40, 85019, 16, 0, 0, 45};
        // categoricFeatures = {"Private", "Doctorate", "Married-civ-spouce", "Prof-specialty", "Husband", "Asian-Pac-Islander", "Male", "nan"};
        //
        // And according to the dataset Person B makes more than 50K a year. Ok, lets try to apply the model to this
        // example.

        System.out.print("\n");

        final float[] personBNumericFeatures = new float[]{40, 85019, 16, 0, 0, 45};
        final String[] personBCategoricFeatures = new String[]{"Private", "Doctorate", "Married-civ-spouce", "Prof-specialty", "Husband", "Asian-Pac-Islander", "Male", "nan"};
        final CatBoostPredictions personBPrediction = adultModel.predict(personBNumericFeatures, personBCategoricFeatures);
        final double personBMakeOver50KProbability = sigmoid(personBPrediction.get(0, 0));
        final boolean personBMakesOver50K = personBMakeOver50KProbability > classificationThreshold;
        System.out.print(String.format("Person B make over 50K a year with probability %f\n", personBMakeOver50KProbability));
        System.out.print(String.format("Person B %s\n", answer(personBMakesOver50K)));

        // There is also a batch interface for model application, e.g. you can apply model to multiple objects at once.
        //
        // NOTE: batch interface is preferable (especially if you are doing highload applications). Time to apply model
        // on multiple object in batch will be less than applying model on each object separately. CatBoost applier uses
        // SIMD to accelerate model applications, also you will pay overhead on JNI call only once for batch
        // application.
        //
        // Let's try to apply the model to Person A and Person B in one call.

        System.out.print("\n");

        final float[][] personsABNumbericFeatures = new float[][]{personANumericFeatures, personBNumericFeatures};
        final String[][] personsABCategoricFeatures = new String[][]{personACategoricFeatures, personBCategoricFeatures};
        final CatBoostPredictions personsABPredictions = adultModel.predict(personsABNumbericFeatures, personsABCategoricFeatures);
        final double[] personsABMakeOver50KProbabilities = new double[]{
                sigmoid(personsABPredictions.get(0, 0)),
                sigmoid(personsABPredictions.get(1, 0))};
        final boolean[] personsABMakeOver50K = new boolean[]{
                personsABMakeOver50KProbabilities[0] > classificationThreshold,
                personsABMakeOver50KProbabilities[1] > classificationThreshold};

        System.out.print("Using batch interface\n");

        // Predictions should be same as above
        System.out.print(String.format("Person A make over 50K a year with probability %f\n", personsABMakeOver50KProbabilities[0]));
        System.out.print(String.format("Person A %s\n", answer(personsABMakeOver50K[0])));
        System.out.print(String.format("Person B make over 50K a year with probability %f\n", personsABMakeOver50KProbabilities[1]));
        System.out.print(String.format("Person B %s\n", answer(personsABMakeOver50K[1])));

        // TODO(yazevnul): add examples with hashed categorical features when `CatBoostModel.hashCategoricalFeatures`
        // will become public.
    }

    public static void main(String[] args) throws CatBoostError, IOException {
        final CatBoost4jPredictionTutorial tutorial = new CatBoost4jPredictionTutorial();
        tutorial.playWithModelForAdultDataset();
    }
}
