# Telegram news style LoRA

**Новый эксперимент:** [перенос подхода style-lora и результаты](experiments/style-transfer-v2/RESULTS.md), [сырые ответы и слепое сравнение](experiments/style-transfer-v2/README.md). Новый адаптер реально обучен, итоговые тексты не редактировались. Допуск по фактам пока не пройден. Ниже сохранена первоначальная сдача и её ограничения.

Одна обученная QLoRA для Qwen3-4B-Instruct-2507, два русскоязычных стиля, пять тестовых новостей в каждом. [Репозиторий](https://github.com/yorik338/telegram-news-style-lora).

Начать с [RESULTS.md](RESULTS.md): выбор модели, гиперпараметры, три сравнения до/после, результаты и готовая команда vLLM для одной GPU 24 GiB / 15 запросов / 4096 токенов.

- [Финансовые посты](outputs_type1.txt), [городские посты](outputs_type2.txt).
- [Обученный адаптер](adapter/), [50 исходных пар](data/original/), [данные обучения](data/).
- [Сырые генерации](evaluation/generation_first_attempts.jsonl), [явные редакторские правки](evaluation/editorial_edits.json), [отложенная проверка](evaluation/holdout_source_audit.json).
- [Linux / WSL запуск](WSL.md), [сервер vLLM](serve_vllm.sh), [клиент](vllm_client.py), [нагрузчик](load_test.py).

**Ограничения:** финальные TXT прошли AI-редакторскую правку (9/10 постов, 23 изменения). Улучшение фактов самой LoRA на holdout не подтверждено: минимум 4 критических поста у базы против 7 у адаптера. Фактическое развёртывание vLLM на 24 GiB и HTTP-нагрузка 15 запросов не проводились. Подробности и сырые данные сохранены.

```bash
python -m pip install -r requirements.txt
python verify_submission.py
python pipeline.py --source news.txt --style 1
python pipeline.py --source news.txt --style 2
```

`news.txt` — UTF-8 источник. Для нового обучения: `python train.py --epochs 1`; результаты записываются отдельно в `training/reproduction/`. База скачивается по закреплённой revision, в репозитории только адаптер. Не публиковать новые ответы без редакторской сверки с источниками.
