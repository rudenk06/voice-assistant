# Голосовой ассистент кафедры

Оффлайн голосовой ассистент для университетской кафедры. Система записывает вопрос, распознаёт речь, ищет сведения в локальной базе документов, формирует краткий ответ и озвучивает его через Piper TTS.

Проект ориентирован на Raspberry Pi 5 с 8 GB RAM, но может запускаться на Linux-компьютере с Python 3.11+.

## Архитектура

```text
Button / Enter
      |
      v
Audio recording
      |
      v
ASR: GigaAM v3 CTC ONNX
      |
      v
RAG retrieval: multilingual-e5-base ONNX + FAISS IndexIDMap + SQLite
      |
      v
Answer generation:
  - direct extraction for contact/email questions
  - local Qwen2.5-1.5B-Instruct GGUF for person, overview, department and other questions
      |
      v
TTS: Piper ru_RU-irina-medium + text normalization
      |
      v
Playback
```

Основной режим активации — GPIO-кнопка или fallback на Enter. Wake word через Vosk есть в коде, но в текущей конфигурации отключён.

## Стек технологий

| Компонент | Технология | Назначение |
| --- | --- | --- |
| ASR | GigaAM v3 CTC, ONNX Runtime | Распознавание русской речи |
| Audio | sounddevice, NumPy | Запись и воспроизведение аудио |
| Embeddings | multilingual-e5-base ONNX, tokenizers | Векторизация запросов и документов |
| Vector search | FAISS IndexIDMap | Поиск ближайших чанков по explicit embedding id |
| Metadata | SQLite | Хранение документов, чанков и embedding id |
| LLM | Qwen2.5-1.5B-Instruct GGUF Q4_K_M, llama-cpp-python | Генерация ответов по найденному контексту |
| TTS | Piper ru_RU-irina-medium ONNX | Синтез русской речи |
| Text normalization | `src/tts/text_normalizer.py` | Подготовка годов, email, доменов и аббревиатур к озвучиванию |

Не все модели используют ONNX Runtime: ASR, embeddings и Piper работают через ONNX, а локальная LLM запускается как GGUF-модель через llama-cpp-python.

## Требования

### Аппаратные

- Raspberry Pi 5 8 GB RAM или Linux-компьютер
- USB-микрофон
- Динамик или аудиовыход
- GPIO-кнопка на пине 17, опционально

### Системные

- Python 3.11+
- `portaudio19-dev`
- `libsndfile1`
- `cmake`
- `build-essential`
- достаточно места для локальных моделей в `data/models/`

## Быстрый старт

```bash
git clone https://github.com/rudenk06/voice-assistant.git
cd voice-assistant

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Для Raspberry Pi также можно использовать установочный скрипт:

```bash
bash scripts/install.sh
```

Скрипт устанавливает системные зависимости, создаёт виртуальное окружение, скачивает доступные модели, индексирует документы и регистрирует systemd-сервис.

## Модели и данные

Модели и индексы не хранятся в git. Ожидаемые пути задаются в `config/assistant.yaml`:

```text
data/models/gigaam-v3-ctc-onnx/
data/models/multilingual-e5-base-onnx/
data/models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf
data/models/piper-ru_RU-irina-medium/
data/index/faiss.index
data/index/chunks.db
```

`scripts/download_models.sh` скачивает ASR и Piper-модель, а также показывает команды для подготовки ONNX-версии multilingual-e5-base. GGUF-модель LLM нужно положить в путь, указанный в конфиге.

Документы базы знаний размещаются в `data/documents/`. Поддерживаются `.txt`, `.pdf` и `.docx`.

## Индексация документов

Ручная индексация:

```bash
source .venv/bin/activate
python3 -m src.rag.indexer
```

Индексатор:

- загружает документы из `data/documents/`;
- разбивает текст на чанки по пустым строкам;
- строит embeddings через multilingual-e5-base ONNX;
- сохраняет FAISS-индекс в `data/index/faiss.index`;
- сохраняет metadata и тексты чанков в SQLite `data/index/chunks.db`.

При запуске ассистента включается `DocumentWatcher`: он polling-ом проверяет `data/documents/` каждые 60 секунд и переиндексирует добавленные или изменённые документы.

## Запуск

```bash
source .venv/bin/activate
python3 -m src.main
```

После запуска нажмите GPIO-кнопку или Enter, задайте вопрос голосом и дождитесь голосового ответа. Основной цикл ассистента находится в `src/main.py`.

## Конфигурация RAG и LLM

Ключевые параметры находятся в `config/assistant.yaml`:

```yaml
rag:
  use_llm: true
  llm_backend: llama_cpp
  llm_model_path: data/models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf
  llm_context_size: 2048
  llm_threads: 3
  llm_max_tokens: 64
  llm_temperature: 0.1
  llm_unload_after_generate: false
  top_k: 2
  contact_top_k: 5
  overview_top_k: 5
  department_top_k: 5
```

Текущая логика генерации гибридная:

- contact/email intent: поиск выполняется с `contact_top_k`, email извлекается напрямую из найденных чанков, LLM не вызывается;
- overview и department intent: retrieval берёт до 5 чанков, затем собирается компактный контекст из найденных фактов, и уже он передаётся в LLM;
- person и other intent: LLM отвечает только по найденному контексту;
- если сведений нет, используется fallback: `В материалах кафедры нет информации по этому вопросу.`

Baseline без LLM сохраняется в коде генератора: при `rag.use_llm: false` ответ формируется шаблонно по лучшему найденному чанку.

## TTS-нормализация

Перед передачей текста в Piper применяется `normalize_for_tts()`. Нормализация не меняет исходный answer в логах, а только строку, которая отправляется в синтез речи.

Обрабатываются:

- email-адреса и домены `spbu.ru`;
- аббревиатуры, включая `СПбГУ`;
- годы вида `в 1989 году`;
- отдельные ручные замены для слов, которые плохо произносятся TTS.

## Производительность

Метрики зависят от устройства, размера модели и типа вопроса. В текущей версии LLM увеличивает задержку по сравнению с retrieval-only режимом, а contact/email вопросы обрабатываются быстрее, потому что LLM для них не вызывается.

Ориентиры по внутренним измерениям проекта:

- ASR latency: около 1623 мс;
- Piper TTS average synthesis: около 0.87 с;
- Piper TTS average RTF: около 0.175.

Для локальных измерений есть скрипты:

```bash
python3 scripts/evaluate_tts.py
python3 scripts/evaluate_rag_llm.py
```

Они не запускают основной цикл ассистента, микрофон или playback.

## Структура проекта

```text
config/
  assistant.yaml          # Основная конфигурация
src/
  main.py                 # Оркестрация полного pipeline
  config.py               # Загрузка YAML-конфига и относительных путей
  asr/
    recognizer.py         # GigaAM v3 CTC ONNX + CTC decode
    wake_word.py          # Опциональный Vosk wake word
  audio/
    recorder.py           # Запись аудио до тишины
    player.py             # Воспроизведение аудио
  hardware/
    button.py             # GPIO-кнопка и Enter fallback
  rag/
    document_loader.py    # Загрузка PDF/DOCX/TXT и chunking
    embedder.py           # multilingual-e5-base ONNX embeddings
    indexer.py            # FAISS IndexIDMap + SQLite
    retriever.py          # Поиск чанков по embedding id
    generator.py          # Intent logic, direct extraction, LLM orchestration
    llm_generator.py      # llama-cpp-python GGUF generator
    watcher.py            # Polling watcher документов
  tts/
    synthesizer.py        # Piper TTS
    text_normalizer.py    # Нормализация текста перед Piper
  utils/
    memory.py             # Логи RSS и GC
    sounds.py             # Системные wav-сигналы
scripts/
  install.sh
  download_models.sh
  index_documents.sh
  evaluate_tts.py
  evaluate_rag_llm.py
benchmarks/
  benchmark_*.py          # Исследовательские benchmark-скрипты
data/
  documents/              # Локальная база документов
  models/                 # Локальные модели, не в git
  index/                  # FAISS/SQLite индекс, не в git
results/                  # Локальные результаты измерений
```

## Лицензия

Код проекта распространяется по лицензии MIT. Лицензии внешних моделей и библиотек нужно учитывать отдельно при распространении собранной системы.
