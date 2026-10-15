# Style transfer v3

Отдельный реально обученный эксперимент: исправленные и компактные train-цели, поддержанные обратные пары, синтетические примеры сохранения условий. Заимствованный подход [style-lora](https://github.com/karansag/style-lora), MIT; [история адаптации](APPROACH.md).

[RESULTS.md](RESULTS.md) содержит фактическое решение и ограничения. [Raw](evaluation/raw_comparison.jsonl) и TXT не подвергались редакторским правкам. [Сверка источников](evaluation/source_audit.json) выполнена AI, не независимым человеком. [Слепое A/B](evaluation/blind_ab.html) подготовлено для человеческой оценки; голосов пока 0.

[Веса](adapter/), [код обучения](train_v3.py), [данные](data/pairs_manifest.json), [замороженный план](evaluation_plan.json), [запуск сервера](serve_vllm.sh). Окружение — из корня репозитория. Проверка: `python experiments/style-transfer-v3/verify_v3.py`. Обработка: `python experiments/style-transfer-v3/rewrite.py --source news.txt --style 1`.
