"""Build the submission report from the preserved actual artifacts."""
import json
from pathlib import Path
from memory_budget import budget

ROOT=Path(__file__).resolve().parent
def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
def rows(name):return [json.loads(s) for s in (ROOT/name).read_text(encoding='utf-8').splitlines() if s.strip()]
def write(name,text):(ROOT/name).write_text(text,encoding='utf-8')

def main():
    epoch=rows('training/epochs.jsonl')[0];artifact=read('training/artifact.json')
    final={(r['id'],r['style']):r for r in rows('evaluation/final_outputs.jsonl')}
    retry_seconds=sum(r['seconds'] for r in rows('evaluation/editorial_retry.jsonl'))
    judge_seconds=sum(r['seconds'] for r in rows('evaluation/calibrated_checks.jsonl'))
    estimates=budget();write('evaluation/memory_estimate_24gib.json',json.dumps(estimates,indent=2))
    # All deployment fields explicitly distinguish a calculation from an actual run.
    write('evaluation/deployment_status.json',json.dumps(dict(actual_gpu='RTX 3070 Ti Laptop 8 GiB',
        vllm_24gib_started=False,concurrent_15_http_test_executed=False,
        load_preflight_only=True,wsl_ubuntu_installed=False,
        checkpoint_reloaded_and_inference_executed=True),indent=2))
    selection=read('relative_selection.json')
    selection.update(calibration='evaluation/calibration_run/summary.json',
        calibrated_benchmark='evaluation/selection_run/summary.json',
        training_artifact='training/artifact.json',training_results='RESULTS.md')
    write('relative_selection.json',json.dumps(selection,ensure_ascii=False,indent=2))
    cal=(ROOT/'evaluation/CALIBRATION.md').read_text(encoding='utf-8')
    cal=cal.replace('(relative_selection.json)','(../relative_selection.json)')
    cal=cal.replace('`runs/nemo-qwen14b-calibrated-v2/`, `runs/nemo-qwen14b-calibrated-benchmark-v1/`',
        '[calibration_run](calibration_run/), [selection_run](selection_run/)')
    cal=cal.replace('[telegram-style-finetune](../telegram-style-finetune/README.md)','[RESULTS.md](../RESULTS.md)')
    write('evaluation/CALIBRATION.md',cal)
    report=f'''# Результаты задания: два стиля Telegram

Решение: одна реально обученная QLoRA для **Qwen3-4B-Instruct-2507**, стиль задаётся в промпте. Получены пять постов каждого стиля: [outputs_type1.txt](outputs_type1.txt), [outputs_type2.txt](outputs_type2.txt). Код, веса адаптера и трассы: [GitHub](https://github.com/yorik338/telegram-news-style-lora).

**Существенное ограничение:** итоговые TXT — результат LoRA, одного повторного прохода по редакторской обратной связи и явной AI-редакторской правки по источнику. В 9 из 10 постов внесено 23 изменения. Это не десять необработанных ответов модели и не доказательство полностью автоматической сохранности фактов. Правки, включая смысловые, записаны в [editorial_edits.json](evaluation/editorial_edits.json); первые ответы и повторные генерации сохранены отдельно. Независимой человеческой оценки нет.

## Соответствие ТЗ

| Требование | Артефакт / результат |
|---|---|
| Два русскоязычных стиля | style=1: аналитический @banki_oil; style=2: городской @moscowach, эмодзи и короткие абзацы |
| 25 пар на стиль | [исходные JSONL](data/original/), всего 50; после курации 38 train + 10 validation + 2 демонстрации |
| Дообучение лёгкой открытой модели | [adapter](adapter/), NF4 + PEFT LoRA, один адаптер на оба стиля |
| Гиперпараметры и шаблон | [config.json](training/config.json), [prompt_template.json](prompt_template.json), [chat_template.jinja](adapter/chat_template.jinja) |
| Пять новостей в двух стилях | два TXT выше; исходники [test_inputs.jsonl](data/test_inputs.jsonl) |
| 2–3 сравнения до/после | три полных примера ниже; база без адаптера с наивным промптом |
| Команда vLLM с LoRA и расчёт 24 ГБ / 15 / 4096 | [serve_vllm.sh](serve_vllm.sh), расчёт и порядок профилирования ниже |
| Время и сложности | измеренные этапы и ограничения ниже |
| Репозиторий GitHub | ссылка выше; приватный репозиторий пользователя |

## Почему эта модель и метод

Порядок выбора задан пользователем: факты → стиль → скорость. Строгий первоначальный допуск не прошёл ни один кандидат. Затем пользователь явно разрешил в условиях ограниченного времени выбрать относительного лидера и дообучить его. Это изменение решения зафиксировано в [relative_selection.json](relative_selection.json).

После калибровки Qwen3-14B / NeMo self_check_facts повторно оценены 106 первых ответов: 1.5B — 19/24 одобрений, 3B — 20/24, 4B — 23/24, 7B — 23/24, 8B — 10/10 при незавершённых 10 из 24. Среди полностью проверенных моделей с уже готовыми HF/NF4-весами для обучения 4B дала лучший результат. 7B смешивала языки и сорвала задачу; 7B/8B были доступны локально только в GGUF. Поэтому 4B позволила сразу выполнить QLoRA на доступной карте 8 GiB. Это относительный выбор по доступным данным, а не глобально доказанный победитель.

Один адаптер экономит память и время. Разные системные инструкции и по одному демонстрационному примеру задают стиль; текущий источник S1 отделён от примера EX1. Пример — образец формы, его факты нельзя переносить в новую новость. Фактичность означает поддержку смысловых утверждений источником: допустимы перефразирование и прямые арифметические следствия, но запрещены новые причины, участники, даты, результаты, изменение знаменателя, направления платежа и статуса предложения. Это проверка соответствия предоставленному источнику, а не доказательство достоверности самого источника внешними данными.

## Обучение и воспроизводимость

База: `unsloth/Qwen3-4B-Instruct-2507-bnb-4bit`, revision `f12db89cd5156e090618dded9b4367f23f8f3b33`. Хеш весов базы записан в [model_lock.json](model_lock.json); большие базовые веса скачиваются с Hugging Face и не включены в Git.

| Параметр | Значение |
|---|---|
| Квантизация / вычисления | NF4, FP16, замороженная база |
| LoRA | rank 8, alpha 16, dropout 0.05 |
| Модули | q/k/v/o_proj, gate/up/down_proj |
| Обучаемые параметры | 16 515 072 |
| Данные | 38 train, 10 validation; 2 промпт-демо исключены из обоих наборов |
| Оптимизатор | AdamW, lr 1e-4, weight decay 0.01, clip norm 1 |
| Эпохи / шаги | 1 / 10; microbatch 1, accumulation 4 |
| Расписание | linear, warmup 1 step, seed 42 |
| Память | gradient checkpointing, loss только на токенах assistant включая EOS |
| Контекст train | без усечения источников, максимум 1172 токена |
| Генерация | temperature 0.6, top_p 0.95, top_k 20, seed 42, максимум 512 новых токенов |

Исходные ответы датасета содержали неподтверждённые добавления. [Курация](training/curation_edits.jsonl) исправляет их по источникам и сохраняет исходные варианты. Связанные события разделены по group_id; пересечений train/validation нет. Два prompt-демо зарезервированы до обучения.

Assistant-only loss с `logits_to_keep` сверена с полной masked-loss: абсолютная разница 1.19e-7. В [training/epochs.jsonl](training/epochs.jsonl) validation token loss снизилась с **{epoch['base_validation_token_loss']:.6f} до {epoch['validation_token_loss']:.6f}** (примерно 28%). Это снижение ошибки предсказания токенов, не метрика истинности. Пиковая зарезервированная память **{epoch['peak_reserved_bytes']/2**30:.2f} GiB**, реально проверены backward и optimizer step. Сохранённый адаптер повторно загружен для генерации: {artifact['size_bytes']:,} байт, SHA-256 `{artifact['sha256']}`. Веса содержат 504 конечных тензора.

Установка для native CUDA / HF:

```bash
python -m venv .venv
# Активировать .venv способом своей ОС
python -m pip install -r requirements.txt
python verify_submission.py
python pipeline.py --source news.txt --style 1
python pipeline.py --source news.txt --style 2
python train.py --epochs 1
```

`news.txt` — UTF-8 текст собственного источника. Последняя команда воспроизводит обучение в новой `training/reproduction/`, не заменяя сданный адаптер; нужен доступ к закреплённой базе. `--base-path` позволяет использовать уже скачанный каталог. [training/original_train.py](training/original_train.py) — точный снимок выполненного ранее запуска с прежней структурой workspace; `train.py` — переносимая версия для этого репозитория. Битовая идентичность генерации между HF и vLLM не гарантируется.

## Проверка фактов и независимая от обучения оценка

Судья: Qwen3-14B-Q4_K_M через реальное действие NeMo `self_check_facts`. Окончательная калибровка: **35/37 совпадений**, два ошибочных одобрения, без невалидных вердиктов; 37 контролей и AI-метки уже использовались ранее, поэтому это повторная калибровка, не независимая оценка. GBNF ограничивает только форму `yes/no`. [Калибровка и ограничения](evaluation/CALIBRATION.md), [37 трасс](evaluation/calibration_run/results.jsonl), [106 трасс бенчмарка](evaluation/selection_run/results.jsonl).

После фиксации адаптера и промпта открыты восемь синтетических отложенных источников. На каждом получены оба стиля для базы с полным промптом и для той же базы с LoRA: **16 против 16 постов**, без правок и дополнительных попыток. [Манифест фиксации](evaluation/generation_manifest.json), [сырые ответы](evaluation/final_holdout.jsonl), [сопоставление с источниками](evaluation/holdout_source_audit.json).

AI-сверка документирует минимум **4 критических поста у базы и 7 у LoRA**. Это нижние границы найденных ошибок, а не исчерпывающая фактическая точность. Например, LoRA меняет медианную цену на среднюю, отсутствие данных о годовом изменении — на отсутствие изменения, расход тепла — на денежные расходы. Улучшение фактов после дообучения **не подтверждено**; holdout не редактировался и не использовался для перенастройки.

Повторная проверка судьёй одобрила все 32 holdout-поста и все десять первых тестовых генераций. Следовательно, он пропустил все 11 документированных критических holdout-постов. Даже 35/37 на калибровке не дают оснований автоматически публиковать ответы. Итоговые десять тестовых постов прошли отдельную AI-редакторскую сверку; её вмешательство открыто отражено в артефактах.

Порядок этапов для заданных пяти новостей: [первый проход](evaluation/generation_first_attempts.jsonl) → [обратная связь](evaluation/editor_feedback.json) → [одна повторная генерация](evaluation/editorial_retry.jsonl) → [23 явные правки](evaluation/editorial_edits.json) → [итоговые записи](evaluation/final_outputs.jsonl). `release_reviewed.py` воспроизводит именно записанные правки для этих тестов, а не является общим автоматическим фактчекером. Для новых новостей остаётся необходимой редакторская проверка по источнику.

## До / после: три фактических примера

«До» — база без адаптера с наивным промптом из `pipeline.messages(..., naive=True)`, те же настройки декодирования. «После» — финальное решение с полным промптом, адаптером, повторной генерацией и AI-правками. Это сравнение конечного процесса с наивной базой; улучшения нельзя приписать только LoRA. Изолированное сравнение LoRA приведено выше и имеет отрицательный результат. Первое необработанное «после» также сохранено в [naive_before_after.jsonl](evaluation/naive_before_after.jsonl).

'''
    notes={1:'Аналитический стиль: сжат длинный официальный текст, сохранены 13 банков, 0,3%, сравнение до 2–3% и атрибуция.',
        3:'Городской стиль: убраны придуманные названия служб доставки, сохранена возможность роста и направление выплаты курьерам.',
        4:'Аналитический стиль: убраны неподтверждённые цели технических систем; сохранены наземный характер симуляции, шесть участников и 365 суток.'}
    for case in rows('evaluation/naive_before_after.jsonl'):
        k=(case['id'],case['style'])
        report+=f"### Новость {k[0]}, стиль {k[1]}\n\n{notes[k[0]]}\n\nДо:\n\n```text\n{case['before']['response']}\n```\n\nПосле:\n\n```text\n{final[k]['response']}\n```\n\n"
    report+=f'''## Одна GPU 24 GiB: готовая команда vLLM

Отдельное Linux/WSL2-окружение с `requirements-vllm.txt` (vLLM 0.11.0). База заранее квантована BNB/NF4; LoRA rank 8 загружается отдельно. Оба стиля используют одно API-имя `telegram-style`, различаются сообщения промпта. Судья 14B не размещается одновременно на той же GPU.

```bash
python -m pip install -r requirements-vllm.txt
export VLLM_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
python -m vllm.entrypoints.openai.api_server \\
  --model unsloth/Qwen3-4B-Instruct-2507-bnb-4bit \\
  --revision f12db89cd5156e090618dded9b4367f23f8f3b33 \\
  --tokenizer-revision f12db89cd5156e090618dded9b4367f23f8f3b33 \\
  --served-model-name telegram-base \\
  --host 127.0.0.1 --port 8000 --api-key "$VLLM_API_KEY" \\
  --dtype half --quantization bitsandbytes --kv-cache-dtype auto \\
  --tensor-parallel-size 1 --gpu-memory-utilization 0.85 \\
  --max-model-len 4096 --max-num-seqs 15 --max-num-batched-tokens 1024 \\
  --enable-chunked-prefill --enforce-eager --generation-config vllm \\
  --enable-lora --max-loras 1 --max-cpu-loras 1 --max-lora-rank 8 \\
  --lora-modules "telegram-style=$PWD/adapter"
```

Та же команда находится в `serve_vllm.sh`, который вычисляет абсолютный путь к адаптеру и допускает overrides памяти/параллельности. [Официальные флаги vLLM](https://docs.vllm.ai/en/v0.11.0/configuration/engine_args.html), [LoRA](https://docs.vllm.ai/en/v0.11.0/features/lora.html), [BitsAndBytes](https://docs.vllm.ai/en/v0.11.0/features/quantization/bnb.html).

`max_model_len=4096` — общий контекст: вход **вместе с** выходом. Для выхода 512 токенов вход с chat template ограничен 3584. Если нужны именно 4096 входных токенов плюс 512 выходных, установить 4608 и пересчитать память: KV для 15 запросов возрастёт до 9.4921875 GiB. Переполнение в клиенте HF отклоняется, источник не обрезается.

`gpu_memory_utilization=0.85` задаёт бюджет executor: 24×0.85 = **20.4 GiB**, оставляет 3.6 GiB вне него. Это не защита от других процессов. Стартовое ограничение: `u ≤ (T − O − R)/T`, где T — реальная ёмкость карты, O — другие процессы, R — резерв (здесь минимум 2 GiB). Сначала проверить `nvidia-smi`. Используются eager и chunked prefill с небольшим batch cap, чтобы ограничить графы и пиковые активации; `max_num_seqs=15` ограничивает исполняемые последовательности, но само по себе не резервирует нужный KV.

Расчёт из конфигурации модели: 36 слоёв, 8 KV-голов, **явный head_dim=128**, FP16 по 2 байта. K и V на токен: `2×36×8×128×2 = 147456 байт` = 144 KiB. Один полный контекст 4096: **576 MiB**; пятнадцать: **8.4375 GiB** или 61 440 KV-токенов. Не следует вычислять head_dim как 2560/32: у этой модели он явно задан и отличается.

При **предположении**, что измеренные веса + LoRA + activation/workspace уложатся в 5.5 GiB, доступно 20.4−5.5 = 14.9 GiB KV; расчёт даёт около 108 498 токенов и запас относительно 61 440. **5.5 GiB — placeholder, не измерение vLLM.** Размер файла базы 2.47 GiB и HF-пик обучения не заменяют профилирование runtime. [Калькулятор](memory_budget.py), [расчёт](evaluation/memory_estimate_24gib.json).

Порядок защиты от OOM на реальной 24 GiB карте:

1. Запустить сервер, записать его фактическую ёмкость GPU KV cache. Она должна быть ≥61 440 токенов для пятнадцати одновременно полных контекстов. Учесть занятость карты другими процессами и запас на колебания.
2. Если KV меньше, освободить ресурсы или в пределах формулы увеличить utilisation; иначе снизить параллельность/контекст и явно признать, что режим 15×4096 не достигнут. При startup OOM сначала уменьшить batch cap, проверить пики workspace; не повышать utilisation вслепую.
3. После `/health` проверить адаптер в `/v1/models` и один запрос; затем выполнить нагрузку ниже. Проверить 45/45 ответов, отсутствие OOM/preemption, фактический выход 512 токенов, длины контекстов и задержки. Входящие запросы сверх admission limit ставить в ограниченную очередь.

```bash
python vllm_client.py --source news.txt --style 1
python load_test.py --prepare-only
python load_test.py --concurrency 15 --rounds 3 > load_result.json
```

Нагрузчик использует разные префиксы и почти полный контекст, чтобы общий prefix cache не скрыл максимальное потребление. Подготовка пятнадцати запросов реально проверена: по 3568 входных +512 выходных =4080 токенов. **HTTP-нагрузка 15 запросов и сам vLLM на 24 GiB здесь не запускались**: доступна только RTX 3070 Ti Laptop 8 GiB, где один KV для 15×4096 больше всей физической памяти. [Статус](evaluation/deployment_status.json), [preflight](evaluation/load_preflight.json), [инструкции WSL](WSL.md).

## Измеренное время и сложности

| Этап | Измеренное время |
|---|---:|
| Пробный optimizer step на самом длинном примере, обновление отброшено | 2.88 с |
| Одна эпоха и validation | {epoch['seconds']:.2f} с |
| Первый тестовый проход, три naive сравнения и 32 holdout-ответа, весь запуск | {read('evaluation/generation_duration.json')['seconds']:.2f} с |
| Десять повторных генераций по обратной связи, сумма inference | {retry_seconds:.2f} с |
| 55 окончательных проверок сильным судьёй, сумма времени запросов | {judge_seconds:.2f} с |

Это длительности разных этапов, а не полный учёт активного рабочего времени. Работа шла 12–13 октября с прерываниями; точное общее число часов не записано, соответствие ориентиру 3–4 часа не заявляется. Скачивания, подготовка, просмотр и правки в таблицу не включены.

Основные сложности: неисправный глобальный Python/Torch решён изолированным Python 3.12 CUDA-окружением; ограничение 8 GiB потребовало NF4, checkpointing и экономии logits; исходные стилистические ответы содержали добавленные факты; судья выдавал обрезанные объяснения, затем был ограничен форматом yes/no, но продолжил пропускать смысловые ошибки. Одной эпохи достаточно для реально обученного артефакта и снижения validation loss, однако недостаточно для доказанного улучшения фактичности. Повторная генерация по замечаниям тоже ошибалась, поэтому итоговые тексты исправлены AI-редактором с открытым журналом вмешательств.

## Проверки сдачи

`python verify_submission.py` проверяет целостность весов/данных/замороженных снимков, отсутствие пересечений, все 10 выходов, воспроизводимость редакторских изменений, цитаты негативного holdout-аудита и математический бюджет KV. `python load_test.py --prepare-only` проверяет подготовку контекстов, без HTTP. Компиляция Python и синтаксис shell проверены; сохранённый адаптер действительно работал в native HF inference. Эти проверки не заменяют не проведённое развёртывание vLLM и независимую человеческую оценку.
'''
    write('RESULTS.md',report)
    write('README.md','''# Telegram news style LoRA

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
''')

if __name__=='__main__':main()
