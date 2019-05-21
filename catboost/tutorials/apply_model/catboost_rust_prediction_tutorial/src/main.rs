// Bring catboost module into the scope
use catboost;

fn sigmoid(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}

fn answer(makes_over_50k_a_year: bool) -> &'static str {
    if makes_over_50k_a_year {
        "makes over 50K a year"
    } else {
        "doesn't make over 50K a year"
    }
}

fn main() {
    // Load "adult.cbm" model that we trained withing Jupyter Notebook
    let model_path = "adult.cbm";
    let model = catboost::Model::load(model_path).unwrap();

    // You can also try to load your own model just replace "adult.cbm" with path to your model that classifies data
    // from UCI Adult Dataset.

    println!("Adult dataset model metainformation\n");

    println!("tree count: {}", model.get_tree_count());

    // In our case we were solving a binary classification problem (weather person makes over 50K a year), so the
    // dimension of the prediction will be 1, it will return probability of the object to belong to the positive
    // class; in our case we had two classed encoded as "<=50K" and ">50K", during data preprocessing (see
    // `get_fixed_adult()` in Notebook) we encoded "<=50K" as 0 and ">50K" as 1, so that ">50K" became a positive
    // class. Probability of the negative class ("<=50K") can be easily deduced as (1-p) where p is a probability of
    // positive class.
    //
    // For most of cases prediction dimension will be 1 (for regression and for ranking), it can be N for cases of
    // multiclassification, where N is a number of classes.
    println!("prediction dimension: {}", model.get_dimensions_count());

    println!("numeric feature count: {}", model.get_float_features_count());

    println!("categoric feature count: {}", model.get_cat_features_count());

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

    println!();

    let person_a_numeric_features = vec![25., 226_802., 7., 0., 0., 40.];
    let person_a_categoric_features = vec![
        String::from("Private"),
        String::from("11th"),
        String::from("Never-married"),
        String::from("Machine-op-inspct"),
        String::from("Own-child"),
        String::from("Black"),
        String::from("Male"),
        String::from("United-States"),
    ];
    let person_a_prediction = model
        .calc_model_prediction(
            vec![person_a_numeric_features.clone()],
            vec![person_a_categoric_features.clone()],
        )
        .unwrap();

    // Since we made prediction only for one person and prediction dimension is 1, proability of person A make
    // over 50K will have index 0 in `person_a_prediction`.
    //
    // CatBoost doesn't compute "probability", to turn CatBoost prediction into a probability we'll need to apply
    // sigmoid function.
    let person_a_makes_over_50k_probability = sigmoid(person_a_prediction[0]);
    println!(
        "Person A make over 50K a year with probability {}",
        person_a_makes_over_50k_probability
    );

    // When we were training CatBoost we used a default classification threshold for AUC which is equal to 0.5,
    // this means that our formula is optimized for this threashold, though we may change threshold to optimize some
    // other metric on a different dataset, but we won't do it in this tutorial.
    let classification_threshold = 0.5;

    let person_a_makes_over_50k = person_a_makes_over_50k_probability > classification_threshold;
    println!("Person A {}", answer(person_a_makes_over_50k));

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

    println!();

    let person_b_numeric_features = vec![40., 85019., 16., 0., 0., 45.];
    let person_b_categoric_features = vec![
        String::from("Private"),
        String::from("Doctorate"),
        String::from("Married-civ-spouce"),
        String::from("Prof-specialty"),
        String::from("Husband"),
        String::from("Asian-Pac-Islander"),
        String::from("Male"),
        String::from("nan"),
    ];
    let person_b_prediction = model
        .calc_model_prediction(
            vec![person_b_numeric_features.clone()],
            vec![person_b_categoric_features.clone()],
        )
        .unwrap();
    let person_b_makes_over_50k_probability = sigmoid(person_b_prediction[0]);
    let person_b_makes_over_50k = person_b_makes_over_50k_probability > classification_threshold;
    println!(
        "Person B make over 50K a year with probability {}",
        person_b_makes_over_50k_probability
    );
    println!("Person B {}", answer(person_b_makes_over_50k));

    // Let's try to apply the model to Person A and Person B in one call.

    println!();

    let persons_ab_numberic_features = vec![person_a_numeric_features, person_b_numeric_features];
    let persons_ab_categoric_features = vec![person_a_categoric_features, person_b_categoric_features];
    let persons_ab_predictions = model
        .calc_model_prediction(persons_ab_numberic_features, persons_ab_categoric_features)
        .unwrap();
    let persons_ab_make_over_50k_probabilities =
        vec![sigmoid(persons_ab_predictions[0]), sigmoid(persons_ab_predictions[1])];
    let persons_ab_make_over_50k = vec![
        persons_ab_make_over_50k_probabilities[0] > classification_threshold,
        persons_ab_make_over_50k_probabilities[1] > classification_threshold,
    ];

    println!("Using batch interface");

    // Predictions should be same as above
    println!(
        "Person A make over 50K a year with probability {}",
        persons_ab_make_over_50k_probabilities[0]
    );
    println!("Person A {}", answer(persons_ab_make_over_50k[0]));
    println!(
        "Person B make over 50K a year with probability {}",
        persons_ab_make_over_50k_probabilities[1]
    );
    println!("Person B {}", answer(persons_ab_make_over_50k[1]));
}
