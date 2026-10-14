"""Report actual v2 artifacts; retain negative results and distinguish AI/human review."""
from common import ROOT,read,rows,dump,sha

def main():
    config=read(ROOT/'training/config.json');artifact=read(ROOT/'training/artifact.json')
    audit=read(ROOT/'evaluation/source_audit.json');diag=read(ROOT/'evaluation/style_diagnostics.json')
    pairs=read(ROOT/'data/pairs_manifest.json');epochs=rows(ROOT/'training/epochs.jsonl')
    review=read(ROOT/'data/neutralization_review.json');selection=read(ROOT/'evaluation/selection.json')
    fact_rows='\n'.join(f"| {split} | {variant} | {s['posts']} | {s['critical_posts']} | {s['coverage_concerns']} |" for split,vs in audit['summary'].items() for variant,s in vs.items())
    metrics='\n'.join(f"| {name} | {m['posts']} | {m['mean_words']:.1f} | {m['mean_training_8gram_overlap']:.4f} | {m['truncated']} |" for name,m in diag['metrics'].items())
    lines='\n'.join(f"| {e['epoch']} | {e['train_loss']:.6f} | {e['validation_token_loss']:.6f} | {e['optimizer_steps']} | {e['peak_reserved_bytes']/2**30:.2f} |" for e in epochs)
    text=f'''# Новый эксперимент: нейтральный текст → стиль

Перенесён подход [karansag/style-lora](https://github.com/karansag/style-lora), commit `{read(ROOT/'upstream_manifest.json')['commit']}`, MIT. Выбор проекта и изменения подробно описаны в [APPROACH.md](APPROACH.md). [Неизменённые исходники](upstream/style-lora/) и лицензия сохранены. Это наиболее подходящий из рассмотренных референсов для нашей задачи, а не доказанный универсальный победитель.

**Новые итоговые TXT содержат первые сырые ответы после исправления технического формата чата. Редакторских правок и повторной генерации по фактологическим замечаниям нет.** До исправления получено 14 ответов на предоставленные источники; они сохранены в [initial_non_autocast](evaluation/initial_non_autocast/). Свежие синтетические источники до исправления не генерировались. Старый эксперимент и его отредактированные TXT сохранены отдельно.

## Исправление инференса

Полный assistant-chat локального токенизатора автоматически добавлял пустой блок `<think>`, а generation header его не добавлял. Поэтому хорошая teacher-forced loss сочеталась с пустыми генерациями адаптера. Проверка precision проблему не устранила. Исправлен только префикс генерации: добавлены четыре фиксированных служебных токена токенизатора, независимо от текста ответа. Веса, промпт и фактическое содержание не менялись; повторного обучения не было. [Диагностика](evaluation/chat_format_diagnostic.json), [описание исправления](evaluation/runtime_fix.json), [точный training-код](training/source_snapshots/). Для vLLM тот же формат задаёт `inference_chat_template.jinja`.

## Данные

38 ранее AI-очищенных стилевых референсов обучения относятся к {config['unique_train_groups']} группам событий. 10 референсов — валидация, 2 — отдельные промпт-демонстрации. Для нейтрализации использована закреплённая Qwen3-4B без адаптера, greedy. Ответы нейтрализатора сохранены в [neutralization_raw.jsonl](data/neutralization_raw.jsonl).

Проверено 48 нейтрализаций: принято {review['summary']['accepted']}, исключено {review['summary']['rejected']}. Среди ошибок были выдуманная причинность, изменённый охват статистики, утрата условия и повреждённые сущности. Решения записаны в [neutralization_review.json](data/neutralization_review.json). Проверка выполнена AI, независимой человеческой курации не было. Подписи канала удалены из нейтральных входов как метаданные; raw сохранён.

В train {config['train_rows']} строк: 38 исходных пар и 23 принятых обратных пары. Это дополнительные представления исходных референсов, а не {config['train_rows']} независимых новостей. Синтетические варианты валидации не участвуют в выборе эпохи: используется loss на десяти исходных входах. [Манифест данных](data/pairs_manifest.json).

## Реальное обучение

База: `unsloth/Qwen3-4B-Instruct-2507-bnb-4bit`, revision `f12db89cd5156e090618dded9b4367f23f8f3b33`. Backend — native CUDA Transformers/PEFT вместо установки Unsloth. Один адаптер на два стиля; общий код промпта [common.py](common.py) используется в обучении и инференсе.

Rank 16, alpha 32, dropout 0.05; q/k/v/o/gate/up/down projections; NF4, FP16 autocast, gradient checkpointing. AdamW lr 2e-4, decay 0.01, cosine, warmup {config['warmup_steps']} шага; 2 эпохи, microbatch 1, accumulation 4, seed 42. Trainable parameters: {config['trainable_parameters']:,}. Loss только на assistant; эквивалентность полной masked-loss проверена, расхождение {config['loss_equivalence_abs_difference']:.3g}. Максимальная обучающая длина {config['max_actual_tokens']} токенов, источники не обрезались.

| Эпоха | Train loss | Validation token loss | Накопленные шаги | Peak reserved GiB |
|---|---:|---:|---:|---:|
{lines}

Loss базы на этих новых целях: {artifact['base_validation_token_loss']:.6f}. Выбрана эпоха {read(ROOT/'training/best_checkpoint.json')['epoch']} по минимальному validation loss. Старый loss из прежнего эксперимента напрямую не сравнивается с новым: цели и промпт изменились.

Обучение заняло {artifact['seconds']:.2f} с после начальной валидации. Сумма времени 48 нейтрализаций: {sum(r['seconds'] for r in rows(ROOT/'data/neutralization_raw.jsonl')):.2f} с; сумма генераций сравнения: {read(ROOT/'evaluation/duration.json')['generation_seconds']:.2f} с. Полное активное время подготовки/анализа не измерялось.

Выбранный адаптер экспортирован в FP16: {artifact['bytes']:,} байт, SHA-256 `{artifact['sha256']}`. Максимальная абсолютная ошибка округления параметров: {artifact['max_abs_export_rounding']:.3g}. Экспорт действительно перезагружен с диска до оценки; 504 тензора проверены на конечность. [Конфигурация](training/config.json), [эпохи](training/epochs.jsonl), [артефакт](training/artifact.json).

## Сравнение сырых генераций

Та же база, одинаковый новый промпт и greedy-декодирование, максимум 384 новых токена. Пять предоставленных источников × два стиля × два варианта =20 ответов. Ещё шесть новых синтетических источников × два стиля × два варианта =24 ответа. Всего 44. Новый регрессионный набор создан AI по ранее известным типам ошибок; это не внешний независимый benchmark. Он зафиксирован до обучения и не использован для перенастройки. [План](evaluation_plan.json), [фиксация](evaluation/generation_manifest.json), [все ответы исправленного запуска](evaluation/raw_comparison.jsonl). Инференс выполнялся партиями по два стиля; `seconds` — время партии, разделённое на два, а не latency одиночного запроса. Одновременная нагрузка CS2 на ту же GPU делает этот запуск непригодным для сравнения абсолютной скорости.

Факты и существенные условия сверены AI с источниками. Это предварительная сверка, не калиброванный балл судьи; пограничные трактовки требуют человеческого решения. Проверяется соответствие предоставленному источнику, а не внешняя истинность самого источника. Записаны конкретные цитаты, предупреждения и пропуски, без гарантии отсутствия всех ошибок.

| Набор | Вариант | Постов | Постов с документированными критическими ошибками | Постов с замечаниями к полноте |
|---|---|---:|---:|---:|
{fact_rows}

[Смысловая сверка и цитаты](evaluation/source_audit.json). **Решение:** {selection['decision_ru']}

Три проверяемых сравнения: в новости о доставке финансовая база пишет «сборы перекладываются на курьеров», LoRA — «перейдут доставщикам», исправляя направление денег, но меняя уже сделанное объявление на будущее. В финансовой новости SIRIUS база расширяет отсутствие прямого физического/визуального контакта до всех контактов; LoRA сохраняет ограничения, почти копируя источник. В городской новости о маленьких квартирах оба варианта выдумывают «самую популярную категорию» и действующий запрет — этот пример показывает оставшуюся ошибку, а не улучшение. Полные тексты доступны в raw и TXT.

## Стиль

[Слепая A/B-страница](evaluation/blind_ab.html) содержит 22 пары с перемешанными сторонами. Названия вариантов находятся в отдельном ключе; голоса можно скачать из страницы. На момент отчёта человеческих голосов **0**. Сравнение стиля человеком не подменено метрикой loss или выдуманным результатом судьи.

| Вариант | Постов | Среднее число слов | Средняя доля совпадающих обучающих 8-грамм | Оборвано |
|---|---:|---:|---:|---:|
{metrics}

Подпись канала исключена из 8-грамм. Эти показатели — диагностика длины и копирования, а не доказательство стиля/фактов. Английский embedding AUC из upstream не использован как готовая метрика русского стиля. [Диагностика](evaluation/style_diagnostics.json).

## Где посмотреть и как запустить

Сырые пять постов на стиль для нового адаптера: [тип 1](evaluation/lora_v2/outputs_type1.txt), [тип 2](evaluation/lora_v2/outputs_type2.txt). Контрольная база: [тип 1](evaluation/base/outputs_type1.txt), [тип 2](evaluation/base/outputs_type2.txt).

Из корня репозитория, в существующем native CUDA-окружении с корневым `requirements.txt`:

```bash
python experiments/style-transfer-v2/verify_v2.py
python experiments/style-transfer-v2/rewrite.py --source news.txt --style 1
python experiments/style-transfer-v2/rewrite.py --source news.txt --style 2
```

`--base-path` позволяет указать локальный каталог базы. Новый запуск подготовки/обучения защищён от перезаписи существующего эксперимента; для воспроизведения следует создать отдельную копию программ с пустыми каталогами артефактов, сохранив данные и манифесты. Полные FP32 промежуточные веса в `_checkpoints/` не входят в Git/архив; сданный FP16 экспорт проверен.

Linux/vLLM использует отдельное окружение по корневому `requirements-vllm.txt`:

```bash
export VLLM_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
bash experiments/style-transfer-v2/serve_vllm.sh
```

Скрипт загружает `telegram-style-v2` с `max-lora-rank=16` и исправленным `--chat-template`. Для доступной 8 GiB карты стартовые overrides: `GPU_MEMORY_UTILIZATION=0.80`, `MAX_NUM_SEQS=1`, `MAX_BATCHED_TOKENS=512`; это отправная точка профилирования, не обещание совместимости с одновременно работающей игрой. Сервис vLLM и нагрузка 15×4096 здесь не запускались; расчёт KV для архитектуры остаётся 8.4375 GiB на пятнадцать полных FP16-контекстов, а фактический runtime нужно профилировать. Клиент [vllm_client.py](vllm_client.py) использует тот же новый промпт. Ключ в другом терминале должен совпадать с ключом сервера.

## Ограничения

Маленький корпус, AI-курация, один эксперимент без поиска гиперпараметров, отсутствие человеческого A/B и отсутствие фактического vLLM load test ограничивают выводы. Новый способ заимствован и реально выполнен; его результат не выдаётся за доказанную промышленную готовность. Исходники и предыдущие отрицательные результаты сохранены.
'''
    (ROOT/'RESULTS.md').write_text(text,encoding='utf-8')
    (ROOT/'README.md').write_text('''# Style transfer v2 — перенос подхода style-lora

Реально выполненный отдельный эксперимент по [karansag/style-lora](https://github.com/karansag/style-lora), MIT: нейтрализация референсов → обучение обратному преобразованию → сравнение базы/LoRA → слепая A/B-оценка.

[RESULTS.md](RESULTS.md) содержит фактические результаты и ограничения. [APPROACH.md](APPROACH.md) объясняет выбор проекта и адаптацию. Скопированные исходники и лицензия: [upstream](upstream/style-lora/), точный commit: [манифест](upstream_manifest.json).

Новые TXT — первые сырые ответы после исправления технического формата чата, без AI-редакторских правок. 14 прежних технических попыток сохранены в `evaluation/initial_non_autocast/`; свежие источники до исправления не генерировались.

Допуск по фактам не пройден, улучшение стиля человеком не подтверждено. Адаптер — результат исследовательского эксперимента:

- [LoRA, тип 1](evaluation/lora_v2/outputs_type1.txt), [LoRA, тип 2](evaluation/lora_v2/outputs_type2.txt).
- [База, тип 1](evaluation/base/outputs_type1.txt), [база, тип 2](evaluation/base/outputs_type2.txt).
- [Слепая A/B-страница](evaluation/blind_ab.html), [сверка фактов](evaluation/source_audit.json).
- [Обученный адаптер](adapter/), [код обучения](train_v2.py), [общий промпт](common.py).

Окружение наследует зависимости из корня репозитория. Из корня: `python experiments/style-transfer-v2/verify_v2.py`. Сохранённый адаптер можно использовать через `rewrite.py`, а на Linux — через `serve_vllm.sh` и `vllm_client.py`. Отдельный vLLM-запуск не проверялся.
''',encoding='utf-8')
    print('V2_REPORT_COMPLETE',flush=True)
if __name__=='__main__':main()
