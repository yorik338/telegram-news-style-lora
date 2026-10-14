# Style transfer v2 — перенос подхода style-lora

Реально выполненный отдельный эксперимент по [karansag/style-lora](https://github.com/karansag/style-lora), MIT: нейтрализация референсов → обучение обратному преобразованию → сравнение базы/LoRA → слепая A/B-оценка.

[RESULTS.md](RESULTS.md) содержит фактические результаты и ограничения. [APPROACH.md](APPROACH.md) объясняет выбор проекта и адаптацию. Скопированные исходники и лицензия: [upstream](upstream/style-lora/), точный commit: [манифест](upstream_manifest.json).

Новые TXT — первые сырые ответы после исправления технического формата чата, без AI-редакторских правок. 14 прежних технических попыток сохранены в `evaluation/initial_non_autocast/`; свежие источники до исправления не генерировались.

Допуск по фактам не пройден, улучшение стиля человеком не подтверждено. Адаптер — результат исследовательского эксперимента:

- [LoRA, тип 1](evaluation/lora_v2/outputs_type1.txt), [LoRA, тип 2](evaluation/lora_v2/outputs_type2.txt).
- [База, тип 1](evaluation/base/outputs_type1.txt), [база, тип 2](evaluation/base/outputs_type2.txt).
- [Слепая A/B-страница](evaluation/blind_ab.html), [сверка фактов](evaluation/source_audit.json).
- [Обученный адаптер](adapter/), [код обучения](train_v2.py), [общий промпт](common.py).

Окружение наследует зависимости из корня репозитория. Из корня: `python experiments/style-transfer-v2/verify_v2.py`. Сохранённый адаптер можно использовать через `rewrite.py`, а на Linux — через `serve_vllm.sh` и `vllm_client.py`. Отдельный vLLM-запуск не проверялся.
