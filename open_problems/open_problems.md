1. Добавить в R библиотеку функцию `eval_metrics`
Это часто используемая функция, в R пока что ее нет. Сама функция написана на питоне, нужно обернуть ее в R.
То есть задача заключается в том, чтобы научиться писать на R и вызывать оттуда плюсовый код. Примеров в коде достаточно.

2. Model calculation is not able to read features from stdin
Сейчас применение модели в cmdline режиме можно делать только из файла и писать в файл.
Нас просят сделать возможность читать из stdin, писать в stdout.

3. Если learning_rate == 0, то CatBoost должен бросать `TCatBoostException`.
Добавить в валидацию опций.

4. `baseline` in eval-metrics
Эта функция предполагает, что начальные значения у каждого объекта == 0.
На самом деле это может быть не правдой, если мы обучаемся из бейзлайна.
Поэтому в этой функции надо тоже добавить возможность применяться из бейзлайна.

5. `--name` в режиме eval-metrics
Нужно заполнять поле `name` в json со значениями метрик строкой, переданной через этот параметр.

6. multiple eval sets on GPU
На ЦПУ CatBoost умеет считать метрики для нескольких тестовых датасетов.
Нужно поддержать эту функциональность на GPU.

7. Улучшить `eval_metrics`:
При указании начальной итерации применять всю модель до этой итерации, и уже из этой точки стартовать оценку метрик.
И разрешить `eval_metrics` шаг больше, чем длина ансамбля - обрезать по длине ансамбля.

8. Automatic `class_weights`/`scale_pos_weight` 

9. Allow `skip_train` `loss_function` property in cv method.

10. EvalFeature supports dataframes, not only file

11. Train from file with header and delimiter. Currently it's only possible to train from tsv file without header.

12. Pairwise metrics in `mode_eval_metrics`

13. Rename Custom to UserDefined
for user defined objectives.
ELossFunction::Custom -> PythonUserDefinedPerObject и в туториале в названии и описании кастом убрать тоже

14. Add CatBoost to https://github.com/apple/turicreate

15. `class_names` parameter in Pool constructor

16. Validate CoreML model when doing its conversion (MLTOOLS-2761)

17. Classification loss for noizy data: see NeurIPS 2018 paper  
Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels

18. Add CatBoostClassifier `predict_log_proba` and `decision_function` methods to support better sklearn API
