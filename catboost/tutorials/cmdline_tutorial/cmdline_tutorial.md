# Catboost command line tutorial
### Train classification model

Train classification model with default params in silent mode. 

```
catboost fit --learn-set train.tsv --test-set test.tsv --column-description train.cd  --loss-function Logloss --iterations 1000 --learning-rate 0.03
```

### Train regression model on csv file with header

Train regression model with 1000 trees on comma separated pool with header. If the header specifies something other than the names of the features, the model ignores it.  In this case, names of features were specified in both, column description file and in header of the dataset. If feature names are specified in both places, then only ones from cd-file are used, as you can see in the example:

```
catboost fit --learn-set train.csv --test-set test.csv --column-description train.cd --loss-function RMSE --iterations 1000 --delimiter="," --has-header
```

### Train classification model in verbose mode with multiple error functions

It is possible to calc additional info while learning, such as current error on learn and current plus best error on test error. Remaining and passed time information is also displayed in verbose mode.
Custom loss functions parameter allow to log additional error functions on learn and test for each iteration. The model is saved into ```model.bin``` file by default

```
catboost fit --learn-set train.tsv --test-set test.tsv --column-description train.cd  --loss-function Logloss --iterations 1000 --custom-loss="AUC,Precision,Recall" --learning-rate 0.03 --verbose 10
```
Example test\_error.tsv result:
```
iter	Logloss			AUC				Precision	Recall
0		0.6913617239	0.5				1			0
1		0.6895846977	0.5520833333	1			0
2		0.6881428049	0.5520833333	1			0
3		0.686666081		0.5520833333	1			0
4		0.6851113844	0.5520833333	1			0
```

### Calculate Feature importances

Model was saved into ```model.bin``` file by default, which will be the value of -m option. The output file with data for features analysis is feature_strength.tsv:

```
catboost fstr -m model.bin --input-path train.tsv --cd train.cd --fstr-type PredictionValuesChange -o feature_strength.tsv
```

### Applying the model

Calc model predictions on ```test.tsv```, output will contain: DocId, evaluated class1 probability, target column, columns which contain names and profession, ```#3```. Where ```#3``` is third column in dataset. Results of applying the model  to the eval.tsv file:

```
catboost calc -m model.bin --input-path test.tsv --cd train.cd -o eval.tsv -T 4 --output-columns DocId,Probability,Target,name,profession,#3
```

Example eval.tsv result:

```
DocId	Probability		Target	name		profession	#3
0		0.1263071371	0		Alex		doctor		winter
1		0.1558141636	1		Demid		dentist		summer
2		0.3505153052	1		Valentin	programmer	spring
3		0.5650058751	1		Ivan		doctor		summer
4		0.102227666		0		Ivan		dentist		spring
```

### Random subspace method

To enable rsm for feature bagging use --rsm parameter:
```
catboost fit --learn-set train.tsv --test-set test.tsv --column-description train.cd  --loss-function Logloss --rsm 0.5 --iterations 1000 --learning-rate 0.03
```

### Params file

It is also possible to pass training parameters in a file :
```
{
    "thread_count": 4,
    "loss_function": "Logloss",
    "iterations": 400
}
```
And run the algorithm as follows:
```
catboost fit --learn-set train.tsv --test-set test.tsv --column-description train.cd --params-file params_file.txt --iterations 1000 --learning-rate 0.03
```

If a parameter is specified in two places - in file and as a command line parameter, then the one from command line is used. This example demonstrates this behavior, because iterations are specified in both places