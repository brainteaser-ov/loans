# Автоматическое выявление лексических заимствований

Репозиторий содержит материалы для воспроизведения эксперимента по бинарной классификации пар словоформ вида «форма языка-источника → форма языка-реципиента». Модель решает задачу определения того, является ли заданная пара отношением заимствования.

Данные формируются на основе официальной CLDF-версии World Loanword Database. Исходные таблицы WOLD преобразуются в `dataset.xlsx`, после чего `loan1.py` собирает положительные и отрицательные пары, выполняет разбиение, обучает модель, считает простые модели сравнения и сохраняет таблицы ошибок.

## Состав репозитория

| Файл | Назначение |
|---|---|
| `borrowings.csv` | Таблица WOLD со сведениями о заимствованиях и словах-источниках |
| `forms.csv` | Таблица WOLD со словоформами языков-реципиентов |
| `languages.csv` | Таблица WOLD с языками, названиями, глоттокодами, ареалами и семьями |
| `parameters.csv` | Таблица WOLD со значениями, семантическими полями и Concepticon-идентификаторами |
| `dataset.py` | Сценарий подготовки `dataset.xlsx` из таблиц WOLD |
| `dataset.xlsx` | Итоговая таблица, используемая для обучения |
| `test_pos.txt` | Когнатные пары, используемые в эксперименте как трудные отрицательные примеры |
| `test_neg.txt` | Некогнатные пары, используемые как обычные отрицательные примеры |
| `loan1.py` | Сценарий подготовки пар, обучения модели, расчёта простых моделей сравнения и анализа ошибок |
| `prepared_pairs.csv` | Полный набор пар после сборки положительного и отрицательного классов |
| `train_pairs.csv` | Обучающая часть |
| `val_pairs.csv` | Валидационная часть |
| `test_pairs.csv` | Тестовая часть |
| `run_manifest.json` | Паспорт запуска с числом строк, составом классов и использованными столбцами |
| `results_summary.json` | Сводные результаты простых моделей сравнения и нейросетевых запусков |

## Источник данных

В качестве источника используется CLDF-версия WOLD из репозитория Lexibank.

Официальный источник данных:

```text
https://github.com/lexibank/wold
```

Для формирования `dataset.xlsx` используются четыре таблицы:

* `borrowings.csv`
* `forms.csv`
* `languages.csv`
* `parameters.csv`

Сценарий `dataset.py` связывает записи из `borrowings.csv` с формами, языками и значениями. В итоговую таблицу сохраняются исходные идентификаторы WOLD, включая `ID`, `Target_Form_ID`, `Target_Form_Row_ID`, `Target_WOLD_ID`, `Target_Language_ID_CLDF`, `Target_Glottocode`, `Parameter_ID`, `Concepticon_ID`, `Concepticon_Gloss`.

Такая структура позволяет проследить происхождение строки `dataset.xlsx` до исходных таблиц WOLD.

## Окружение

Рекомендуется использовать Python 3.10 или более новую версию.

Минимальный набор библиотек:

```bash
python -m pip install --upgrade pip
python -m pip install pandas openpyxl numpy scikit-learn torch
```

При работе на компьютере без графического ускорителя можно запускать обучение на `cpu`. При наличии совместимой видеокарты PyTorch может использовать `cuda`.

## Подготовка dataset.xlsx

Перед запуском в корне репозитория должны находиться файлы:

* `borrowings.csv`
* `forms.csv`
* `languages.csv`
* `parameters.csv`

Команда:

```bash
python dataset.py
```

Результат выполнения:

* `dataset.xlsx`

Внутри `dataset.xlsx` создаются листы:

| Лист | Содержание |
|---|---|
| `dataset` | Основная таблица для обучения |
| `summary` | Число строк во входных таблицах, число строк итогового датасета, число несвязанных строк |
| `unmatched` | Строки, которые не удалось связать с исходными таблицами WOLD; лист создаётся при наличии таких строк |

## Формирование dataset.xlsx

 `dataset.py` выполняет следующие действия.

1. Читает `borrowings.csv`, `forms.csv`, `languages.csv`, `parameters.csv`.
2. Разбирает `Target_Form_ID` как сочетание языка, значения и номера формы.
3. Находит соответствующую строку в `forms.csv`.
4. Добавляет сведения о языке из `languages.csv`.
5. Добавляет сведения о значении из `parameters.csv`.
6. Сохраняет итоговую строку в `dataset.xlsx`.

В итоговой таблице положительная пара формируется полями:

| Поле | Содержание |
|---|---|
| `Donor` | Форма языка-источника |
| `Target_Form` | Форма языка-реципиента |
| `Target_Language_Name` | Язык-реципиент |
| `Meaning` | Значение |
| `SemanticField` | Семантическое поле |
| `SemanticCategory` | Семантическая категория |
| `label_loan` | Метка положительного класса при наличии формы источника |

## Подготовка отрицательных примеров

В эксперименте используются два файла с отрицательными парами.

| Файл | Роль в эксперименте |
|---|---|
| `test_pos.txt` | Когнатные пары. 
| `test_neg.txt` | Некогнатные пары.  |

Метки в `test_pos.txt` и `test_neg.txt` используются при чтении файлов. В итоговом наборе обе группы получают `label_loan = 0`.

Код сохраняет тип примера в поле `neg_type`:

* `cognate_external`
* `noncognate_external`

Положительные пары из WOLD получают:

* `label_loan = 1`
* `neg_type = positive`
* `pair_source = wold_positive`

## Запуск эксперимента

Команда для основного запуска:

```bash
python loan1.py   --data dataset.xlsx   --sheet dataset   --cognates test_pos.txt   --noncognates test_neg.txt   --outdir loan_artifacts   --negative-ratio 1.0   --split-mode target_form   --test-size 0.15   --val-size 0.15   --seed 42   --epochs 12   --batch-size 128   --max-src-bytes 64   --max-tgt-bytes 64   --d-model 256   --layers 6   --heads 8   --ff-mult 4   --dropout 0.10   --run-ablations
```

Назначение параметров:

| Параметр | Назначение |
|---|---|
| `--data` | Входной Excel-файл |
| `--sheet` | Лист с таблицей данных |
| `--cognates` | Файл когнатных пар |
| `--noncognates` | Файл некогнатных пар |
| `--outdir` | Каталог для результатов |
| `--negative-ratio` | Число отрицательных пар на одну положительную пару до удаления повторов |
| `--split-mode` | Способ группового разбиения |
| `--seed` | Начальное значение генератора случайных чисел |
| `--run-ablations` | Запуск вариантов модели с отключением компонентов |

## Тест

Для проверки работоспособности без полного обучения можно использовать малую выборку.

```bash
python loan1.py   --data dataset.xlsx   --sheet dataset   --cognates test_pos.txt   --noncognates test_neg.txt   --outdir loan_artifacts_debug   --fast-dev-rows 1000   --epochs 2   --batch-size 64   --device cpu   --run-ablations
```

Запуск нужен для проверки чтения файлов, сборки пар, разбиения, обучения и записи выходных таблиц.

## Расчёт простых моделей сравнения без обучения трансформера

Для получения результатов простых моделей сравнения можно отключить нейросетевое обучение.

```bash
python loan1.py   --data dataset.xlsx   --sheet dataset   --cognates test_pos.txt   --noncognates test_neg.txt   --outdir loan_artifacts_baselines   --skip-neural
```

Используются простые модели сравнения:

* порог по сходству Левенштейна;
* логистическая регрессия на строковых признаках.

Файл результата:

```text
loan_artifacts_baselines/baselines/baseline_metrics.json
```

## Запуск только основной нейросетевой модели

Если нужно обучить полную модель без отключения компонентов, можно убрать `--run-ablations`.

```bash
python loan1.py   --data dataset.xlsx   --sheet dataset   --cognates test_pos.txt   --noncognates test_neg.txt   --outdir loan_artifacts_full   --negative-ratio 1.0   --split-mode target_form   --seed 42   --epochs 12   --batch-size 128
```

## Итоговые файлы loan1.py

После запуска в каталоге `--outdir` создаются следующие файлы.

| Файл или каталог | Содержание |
|---|---|
| `prepared_pairs.csv` | Полный набор пар после сборки и удаления повторов |
| `train_pairs.csv` | Обучающая часть |
| `val_pairs.csv` | Валидационная часть |
| `test_pairs.csv` | Тестовая часть |
| `run_manifest.json` | Паспорт запуска |
| `results_summary.json` | Итоговые метрики |
| `baselines/baseline_metrics.json` | Метрики простых моделей сравнения |
| `full/training_metrics.csv` | Динамика обучения полной модели |
| `full/model_config.json` | Параметры полной модели |
| `full/metadata_maps.json` | Словари метаданных |
| `full/test_predictions.csv` | Предсказания модели на тестовой части |
| `full/confusion_matrix_overall.csv` | Общая матрица ошибок |
| `full/confusion_matrix_by_type.csv` | Матрица ошибок по типам примеров |
| `full/false_positives_top200.csv` | До 200 ложноположительных случаев |
| `full/false_negatives_top200.csv` | До 200 ложноотрицательных случаев |

При запуске с `--run-ablations` создаются каталоги:

| Каталог | Описание |
|---|---|
| `full` | Полная модель с байтовым входом, строковыми признаками и метаданными |
| `no_metadata` | Модель без метаданных |
| `no_string_features` | Модель без строковых признаков |
| `bytes_only` | Модель только с байтовым входом |

## Используемые признаки

Строковые признаки рассчитываются по полям `src_form` и `tgt_form`.

| Признак | Содержание |
|---|---|
| `src_len` | Длина формы источника |
| `tgt_len` | Длина формы реципиента |
| `len_abs_diff` | Абсолютная разница длин |
| `len_ratio_short_long` | Отношение меньшей длины к большей |
| `seqmatcher_ratio` | Сходство последовательностей |
| `levenshtein_similarity` | Сходство на основе расстояния Левенштейна |
| `common_prefix_ratio` | Доля общего префикса |
| `common_suffix_ratio` | Доля общего суффикса |
| `bigram_jaccard` | Индекс Жаккара для биграмм |
| `trigram_jaccard` | Индекс Жаккара для триграмм |
| `first_char_same` | Совпадение первого символа |
| `last_char_same` | Совпадение последнего символа |

## Представление входа модели

Модель использует байтовое представление UTF-8. Каждая словоформа нормализуется через Unicode NFKC, затем кодируется в байты.

Входная последовательность имеет вид:

```text
[CLS] [метаданные] [SEP] байты_источника [SEP] байты_реципиента [SEP]
```

Метаданные включают:

* язык-реципиент;
* семантическое поле;
* семантическую категорию.

При запуске `no_metadata` метаданные не добавляются. При запуске `bytes_only` используются только байты двух словоформ.

## Архитектура модели

`loan1.py` обучает модель `ByteCrossEncoder`.

Основные компоненты:

* вложения байтовых токенов;
* вложения сегментов;
* позиционные вложения;
* кодировщик трансформера;
* классификационный блок с полносвязными слоями;
* бинарная классификация через `BCEWithLogitsLoss`.

Стандартные параметры основного запуска:

| Параметр | Значение |
|---|---|
| `d_model` | 256 |
| `layers` | 6 |
| `heads` | 8 |
| `ff_mult` | 4 |
| `dropout` | 0.10 |
| `max_src_bytes` | 64 |
| `max_tgt_bytes` | 64 |
| `epochs` | 12 |
| `batch_size` | 128 |

## Метрики

Для каждой модели рассчитываются:

* accuracy;
* precision;
* recall;
* F1;
* ROC-AUC;
* loss для нейросетевой модели.

Для простых моделей сравнения порог выбирается по валидационной части, затем метрики считаются на тестовой части.

## Воспроизведение 

Рекомендуемая последовательность действий:

1. Поместить в корень репозитория файлы WOLD `borrowings.csv`, `forms.csv`, `languages.csv`, `parameters.csv`.
2. Запустить подготовку `dataset.xlsx`.

```bash
python dataset.py
```

3. Запустить основной эксперимент.

```bash
python loan1.py   --data dataset.xlsx   --sheet dataset   --cognates test_pos.txt   --noncognates test_neg.txt   --outdir loan_artifacts   --negative-ratio 1.0   --split-mode target_form   --test-size 0.15   --val-size 0.15   --seed 42   --epochs 12   --batch-size 128   --max-src-bytes 64   --max-tgt-bytes 64   --d-model 256   --layers 6   --heads 8   --ff-mult 4   --dropout 0.10   --run-ablations
```

4. Сверить число строк и состав классов в файле:

```text
loan_artifacts/run_manifest.json
```

5. Сверить итоговые метрики в файле:

```text
loan_artifacts/results_summary.json
```

6. Для анализа ошибок использовать файлы:

```text
loan_artifacts/full/confusion_matrix_overall.csv
loan_artifacts/full/confusion_matrix_by_type.csv
loan_artifacts/full/false_positives_top200.csv
loan_artifacts/full/false_negatives_top200.csv
```

## Примечание о балансе классов

Параметр `--negative-ratio 1.0` задаёт число отрицательных пар относительно числа положительных пар на этапе сборки. После удаления повторов число положительных и отрицательных строк может отличаться.

Проверять фактический состав классов по файлу:

```text
run_manifest.json
```

В текущем запуске указаны значения:

| Группа | Число строк |
|---|---|
| Все строки | 38 822 |
| Положительный класс | 18 199 |
| Отрицательный класс | 20 623 |
| Когнатные отрицательные примеры | 10 280 |
| Некогнатные отрицательные примеры | 10 343 |

## Ограничения

Модель классифицирует уже заданные пары словоформ. Полный поиск источника заимствования по словарю не выполняется.

Качество на внешних корпусах зависит от того, как сформированы пары-кандидаты, как представлены формы и насколько новые данные сопоставимы с WOLD.

Пары из `test_pos.txt` и `test_neg.txt` используются как внешние отрицательные примеры. Для полной проверяемости желательно сопровождать эти файлы отдельным описанием источника, правил отбора и числа строк после обработки.

## Цитирование данных

При использовании данных WOLD следует ссылаться на публикацию:

```text
Haspelmath, Martin & Tadmor, Uri (eds.) 2009. World Loanword Database. Leipzig: Max Planck Institute for Evolutionary Anthropology.
```

Также следует указывать репозиторий CLDF-версии WOLD:

```text
https://github.com/lexibank/wold
```



#### Стек
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) 
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

#### Directed by 


[brainteaser-ov 💛](https://github.com/brainteaser-ov)  
