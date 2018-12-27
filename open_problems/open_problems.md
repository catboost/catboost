1. Сохранение бинаризованного датасета
Обучение происходит в несколько этапов. Сначала идет препроцессинг данных, который квантизует факторы.
Хочется иметь возможность сохранять датасет в квантизованном виде. Для этого уже есть прото формат.
Нужно добавить сохранение и чтение.

2. Добавить в R библиотеку функцию `eval_metrics`
Это часто используемая функция, в R пока что ее нет. Сама функция написана на питоне, нужно обернуть ее в R.
То есть задача заключается в том, чтобы научиться писать на R и вызывать оттуда плюсовый код. Примеров в коде достаточно.

3. Model calculation is not able to read features from stdin
Сейчас применение модели в cmdline режиме можно делать только из файла и писать в файл.
Нас просят сделать возможность читать из stdin, писать в stdout.

4. Если learning_rate == 0, то CatBoost должен бросать `TCatBoostException`.
Добавить в валидацию опций.

5. `baseline` in eval-metrics
Эта функция предполагает, что начальные значения у каждого объекта == 0.
На самом деле это может быть не правдой, если мы обучаемся из бейзлайна.
Поэтому в этой функции надо тоже добавить возможность применяться из бейзлайна.

6. `--name` в режиме eval-metrics
Нужно заполнять поле `name` в json со значениями метрик строкой, переданной через этот параметр.

7. Считать AUC в несколько потоков

8. Метрики, использующие confusion matrix, должны при вычислении проходить по датасету 1 раз, сейчас для каждой метрики все считается заново

9. multiple eval sets on GPU
На ЦПУ CatBoost умеет считать метрики для нескольких тестовых датасетов.
Нужно поддержать эту функциональность на GPU.

10. поддержка хидера и делимиторов в эвал фиче
У нас есть режим для оценивания факторов. Этот режим сейчас работает не с произвольным датасетом, а с датасетом в определенном формате, без хидета и с табами. Хочется, чтобы с любым работал.

11. Сделать пример использования CatBoost в Goodle Colaboratory.

12. Улучшить `eval_metrics`:
При указании начальной итерации применять всю модель до этой итерации, и уже из этой точки стартовать оценку метрик.
И разрешить `eval_metrics` шаг больше, чем длина ансамбля - обрезать по длине ансамбля.

13. Добавить новые способы анализа фичей:
Для каждой флотной фичи строить графики, как от нее зависит таргет и прогноз. Добавить подсчет и отрисовку графиков в cmldine, python.

14. Сделать пример использования CatBoost GPU в качестве Kaggle kernel.

15. Задача, в которой надо разбираться:
One-hot encoding in CoreML
Коллеги из CoreML говорят, что он у них поддержан, надо разобраться, как, и добавить в катбуст.
Модели с флотными фичами мы конвертить умеем, модели с one-hot надо сделать по аналогии, но разобраться еще в CoreML.

16. Automatic `class_weights`/`scale_pos_weight` 

17. Написать режим предсказания, в какой лист попадет объект.

18. в cv разрешить свойство функций ошибок `skip_train`

19. shap values on subset
Визуализация для shap values работает только на каком-то не очень большом числе объектов. Надо для визуализации сделать семплирование.

20. EvalFeature supports dataframes, not only file

21. Train from file with header and delimiter. Currently it's only possible to train from tsv file without header.

22. попарные метрики `mode_eval_metrics`

23. Переименовать Custom в UserDefined
for user defined objectives.
ELossFunction::Custom -> PythonUserDefinedPerObject и в туториале в названии и описании кастом убрать тоже

24. model.compare - еще один способ визуального сравнения моделей. Сейчас для сраврения моделей надо использовать отдельный класс виджет. Хотелось бы, чтобы можно было сравнивать при помощи функции compare

25. Load progress from snapshot if `thread_count` or max memory are different.
And if this is the case, then time estimates should be updated: time estimate is made based on previous timings. In case if one of these parameters has changed, previous time estimates become not trustable.

26. AUC for MultiClass

27. Weight in all binarizations.

28. Add CatBoost to https://github.com/apple/turicreate

29. `class_names` parameter in Pool constructor

30. Validate CoreML model when doing its conversion (MLTOOLS-2761)

31. Classification loss for noizy data: see NeurIPS 2018 paper  
Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels
