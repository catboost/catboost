# PMML

The [Predictive Model Markup Language]({{ pmml-v4point3 }}) (PMML) is an XML-based language which provides a way for applications to define statistical and data mining models and to share models between PMML compliant applications.


## {{ input_data__title__peculiarities }} {#specifics}

- {{ product }} exports models to [PMML version {{ pmml-supported-version }}]({{ pmml-v4point3 }}).
- Categorical features must be interpreted as one-hot encoded during the training if present in the training dataset. This can be accomplished by setting the `--one-hot-max-size`/`one_hot_max_size` parameter to a value that is greater than the maximum number of unique categorical feature values among all categorical features in the dataset.
- Multiclassification models are not currently supported.

- Models saved as PMML cannot be currently loaded by {{ product }} libraries/executable. Use this format if the model is intended to be used with external Machine Learning libraries.

- {% include [reusage-common-phrases-native-catboost-format-is-faster](../_includes/work_src/reusage-common-phrases/native-catboost-format-is-faster.md) %}



## Model parameters {#model-parameters}

### {{ common-text__title__reference__input-data }} {#inputs}

Numerical features description format:
```xml
<DataField name="<feature_name>" optype="continuous" dataType="float"/>
```

Categorical features description format:
```xml
<DataField name="<feature_name>" optype="categorical" dataType="string"/>
```

### {{ common-text__title__reference__output-data }} {#outputs}

#### Classification
Only binary classification is currently supported.
```xml
<DataField name="prediction" optype="categorical" dataType="boolean"/>
```

#### Regression

```xml
<DataField name="prediction" optype="continuous" dataType="double"/>
```


## {{ dl--example }} {#examples}

The following examples use the [Python package](python-quickstart.md) for training and the [Java Evaluator API for PMML](https://github.com/jpmml/jpmml-evaluator) for applying the model.

### Binary classification {#binary-classification}

Training:

```python
import catboost
from sklearn import datasets


train_data = datasets.load_breast_cancer()

model = catboost.CatBoostClassifier(loss_function='Logloss')

train_dataset = catboost.Pool(
    train_data.data,
    label=train_data.target,
    feature_names=list(train_data.feature_names)
)
model.fit(train_dataset)

# Save model to PMML format
model.save_model(
    "breast_cancer.pmml",
    format="pmml",
    export_parameters={
        'pmml_copyright': 'my copyright (c)',
        'pmml_description': 'test model for BinaryClassification',
        'pmml_model_version': '1'
    }
)

```

Applying:
```java
package com.mycompany.app;

import java.io.*;
import java.util.*;

import org.xml.sax.SAXException;

import org.dmg.pmml.*;
import org.jpmml.model.*;
import org.jpmml.evaluator.*;


public class App
{
    public static void main(String[] args) throws Exception
    {
        String modelPath = "breast_cancer.pmml";

        Evaluator evaluator = new LoadingModelEvaluatorBuilder()
            .setLocatable(false)
            .setVisitors(new DefaultVisitorBattery())
            //.setOutputFilter(OutputFilters.KEEP_FINAL_RESULTS)
            .load(new File(modelPath))
            .build();

        Map<String, Float> inputDataRecord = new HashMap<String,Float>();
        inputDataRecord.put("mean radius", 17.99f);
        inputDataRecord.put("mean texture", 10.38f);
        inputDataRecord.put("mean perimeter", 122.8f);
        inputDataRecord.put("mean area", 1001.0f);
        inputDataRecord.put("mean smoothness", 0.1184f);
        inputDataRecord.put("mean compactness", 0.2776f);
        inputDataRecord.put("mean concavity", 0.3001f);
        inputDataRecord.put("mean concave points", 0.1471f);
        inputDataRecord.put("mean symmetry", 0.2419f);
        inputDataRecord.put("mean fractal dimension", 0.07871f);
        inputDataRecord.put("radius error", 1.095f);
        inputDataRecord.put("texture error", 0.9053f);
        inputDataRecord.put("perimeter error", 8.589f);
        inputDataRecord.put("area error", 153.4f);
        inputDataRecord.put("smoothness error", 0.006399f);
        inputDataRecord.put("compactness error", 0.04904f);
        inputDataRecord.put("concavity error", 0.05373f);
        inputDataRecord.put("concave points error", 0.01587f);
        inputDataRecord.put("symmetry error", 0.03003f);
        inputDataRecord.put("fractal dimension error", 0.006193f);
        inputDataRecord.put("worst radius", 25.38f);
        inputDataRecord.put("worst texture", 17.33f);
        inputDataRecord.put("worst perimeter", 184.6f);
        inputDataRecord.put("worst area", 2019.0f);
        inputDataRecord.put("worst smoothness", 0.1622f);
        inputDataRecord.put("worst compactness", 0.6656f);
        inputDataRecord.put("worst concavity", 0.7119f);
        inputDataRecord.put("worst concave points", 0.2654f);
        inputDataRecord.put("worst symmetry", 0.4601f);
        inputDataRecord.put("worst fractal dimension", 0.1189f);

        Map<FieldName, FieldValue> arguments = new LinkedHashMap<>();

        List<? extends InputField> inputFields = evaluator.getInputFields();
        for(InputField inputField : inputFields){
            FieldName inputName = inputField.getName();

            Object rawValue = inputDataRecord.get(inputName.getValue());

            // Transforming an arbitrary user-supplied value to a known-good PMML value
            // The user-supplied value is passed through: 1) outlier treatment, 2) missing value treatment, 3) invalid value treatment and 4) type conversion
            FieldValue inputValue = inputField.prepare(rawValue);

            arguments.put(inputName, inputValue);
        }

        Map<FieldName, ?> results = evaluator.evaluate(arguments);

        List<? extends TargetField> targetFields = evaluator.getTargetFields();
        for(TargetField targetField : targetFields){
            FieldName targetName = targetField.getName();

            Object targetValue = results.get(targetName);
            System.out.println(targetName);
            System.out.println(targetValue);
        }

    }
}
```

### Regression {#regression}

Training:

```python
import catboost
from sklearn import datasets


train_data = datasets.load_boston()

model = catboost.CatBoostRegressor()

train_dataset = catboost.Pool(
    train_data.data,
    label=train_data.target,
    feature_names=list(train_data.feature_names)
)
model.fit(train_dataset)

# Save model to PMML format
model.save_model(
    "boston.pmml",
    format="pmml",
    export_parameters={
        'pmml_copyright': 'my copyright (c)',
        'pmml_description': 'test model for Regression',
        'pmml_model_version': '1'
    }
)

```

Applying:

```java
package com.mycompany.app;

import java.io.*;
import java.util.*;

import org.xml.sax.SAXException;

import org.dmg.pmml.*;
import org.jpmml.model.*;
import org.jpmml.evaluator.*;


public class App
{
    public static void main(String[] args) throws Exception
    {
        String modelPath = "boston.pmml";

        Evaluator evaluator = new LoadingModelEvaluatorBuilder()
            .setLocatable(false)
            .setVisitors(new DefaultVisitorBattery())
            //.setOutputFilter(OutputFilters.KEEP_FINAL_RESULTS)
            .load(new File(modelPath))
            .build();

        Map<String, Float> inputDataRecord = new HashMap<String,Float>();
        inputDataRecord.put("CRIM", 0.00632f);
        inputDataRecord.put("ZN", 18.0f);
        inputDataRecord.put("INDUS", 2.31f);
        inputDataRecord.put("CHAS", 0.0f);
        inputDataRecord.put("NOX", 0.538f);
        inputDataRecord.put("RM", 6.575f);
        inputDataRecord.put("AGE", 65.2f);
        inputDataRecord.put("DIS", 4.09f);
        inputDataRecord.put("RAD", 1.0f);
        inputDataRecord.put("TAX", 296.0f);
        inputDataRecord.put("PTRATIO", 15.3f);
        inputDataRecord.put("B", 396.9f);
        inputDataRecord.put("LSTAT", 4.98f);

        Map<FieldName, FieldValue> arguments = new LinkedHashMap<>();

        List<? extends InputField> inputFields = evaluator.getInputFields();
        for(InputField inputField : inputFields){
            FieldName inputName = inputField.getName();

            Object rawValue = inputDataRecord.get(inputName.getValue());

            // Transforming an arbitrary user-supplied value to a known-good PMML value
            // The user-supplied value is passed through: 1) outlier treatment, 2) missing value treatment, 3) invalid value treatment and 4) type conversion
            FieldValue inputValue = inputField.prepare(rawValue);

            arguments.put(inputName, inputValue);
        }

        Map<FieldName, ?> results = evaluator.evaluate(arguments);

        List<? extends TargetField> targetFields = evaluator.getTargetFields();
        for(TargetField targetField : targetFields){
            FieldName targetName = targetField.getName();

            Object targetValue = results.get(targetName);
            System.out.println(targetName);
            System.out.println(targetValue);
        }

    }
}
```
