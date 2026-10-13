# Запуск vLLM на Linux / WSL2

ТЗ требует готовую команду и расчёт для одной GPU с 24 GiB, а не фактическое развёртывание на такой карте. Измерения этой сдачи сделаны в native Windows CUDA на RTX 3070 Ti Laptop 8 GiB. Доступная WSL2-дистрибуция — только остановленная docker-desktop; она не изменялась. Ubuntu и vLLM в WSL не устанавливались.

Если нужен локальный WSL-сервис, сначала установить отдельную Ubuntu, например `wsl --install -d Ubuntu-24.04`, затем проверить `nvidia-smi` внутри неё. Эти действия здесь не выполнялись. Не устанавливать Linux-драйвер NVIDIA поверх WSL-драйвера Windows.

В Linux создать отдельное окружение, скопировать или клонировать этот репозиторий, выполнить:

```bash
python3 -m venv .venv-vllm
source .venv-vllm/bin/activate
python -m pip install -r requirements-vllm.txt
export VLLM_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
bash serve_vllm.sh
```

Для локальной карты 8 GiB начать с меньшей параллельности:

```bash
export GPU_MEMORY_UTILIZATION=0.80
export MAX_NUM_SEQS=1
export MAX_BATCHED_TOKENS=512
bash serve_vllm.sh
```

Это стартовая конфигурация для проверки, не измеренная гарантия. Только KV-кэш пятнадцати полностью заполненных FP16-контекстов 4096 занимает 8.4375 GiB, поэтому на карте 8 GiB одновременно хранить их невозможно даже без весов. Сервер может принимать больше входящих HTTP-запросов в очередь, но это отличается от одновременного исполнения пятнадцати длинных последовательностей.

После готовности `/health` и наличия `telegram-style` в `/v1/models` проверить один запрос через `vllm_client.py`, затем нагрузку. API слушает localhost. Для внешнего доступа нужен отдельный шлюз с аутентификацией и ограничением очереди.

[vLLM GPU installation](https://docs.vllm.ai/en/v0.11.0/getting_started/installation/gpu.html), [CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
