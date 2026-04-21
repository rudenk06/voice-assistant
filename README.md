
# Голосовой ассистент кафедры

Speech-to-speech голосовой ассистент для университетской кафедры на Raspberry Pi 5 (8GB RAM). Полностью оффлайн, open-source.

**Пользователь нажимает кнопку (или говорит "Окей кафедра") → задаёт вопрос голосом → ассистент отвечает голосом**, используя базу знаний из документов кафедры.

## Архитектура

```
Кнопка / Wake word
       │
       ▼
  ASR (GigaAM v3 CTC)    — русская речь → текст
       │
       ▼
  RAG pipeline             — поиск по документам + генерация ответа
  ├─ Embed (multilingual-e5-base ONNX)
  ├─ Search (FAISS)
  └─ Generate (template)
       │
       ▼
  TTS (Piper)             — текст → русская речь
       │
       ▼
  Динамик
```

## Стек технологий

| Компонент | Технология | Формат | Назначение |
|-----------|-----------|--------|------------|
| ASR | GigaAM v3 CTC (int8) | ONNX | Распознавание русской речи |
| Wake word | Vosk keyword spotting | — | Детекция "Окей кафедра" |
| Embeddings | multilingual-e5-base | ONNX | Векторизация текста |
| Vector store | FAISS + SQLite | — | Хранение и поиск чанков |
| Generator | Template-based | — | Формирование ответа |
| TTS | Piper (ru_RU-irina-medium) | ONNX | Синтез русской речи |

**Все модели работают через ONNX Runtime** — без PyTorch в рантайме, оптимально для CPU.

### Производительность (Raspberry Pi 5, 8GB RAM)

| Метрика | Значение |
|---------|----------|
| Время запуска | ~20 сек |
| Время ответа (после прогрева) | ~3.5 сек |
| ASR (распознавание) | ~0.3 сек |
| Embed + поиск | ~1.8 сек |
| TTS (синтез) | ~1.2 сек |
| RAM (пиковое) | ~2.0 GB |

## Требования

### Железо
- Raspberry Pi 5 (8 GB RAM) — или любой Linux-компьютер
- USB-микрофон
- Динамик (USB или 3.5mm jack)
- GPIO-кнопка на пин 17 (опционально — есть fallback на Enter)

### Софт
- Raspberry Pi OS 64-bit (Bookworm) или любой Linux с Python 3.11+
- ~2 GB свободного места на диске

## Быстрый старт

### 1. Клонировать репозиторий
```bash
git clone <repo-url> ~/voice-assistant
cd ~/voice-assistant
```

### 2. Установка (автоматическая)
```bash
bash scripts/install.sh
```

### 2 (альтернатива). Ручная установка
```bash
sudo apt update && sudo apt install -y \
    python3-pip python3-venv \
    portaudio19-dev libsndfile1 \
    cmake build-essential wget unzip

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

bash scripts/download_models.sh
python3 -m src.rag.indexer
```

### 3. Запуск
```bash
source .venv/bin/activate
python3 -m src.main
```

После запуска:
- Нажмите **Enter** (или GPIO-кнопку) → говорите → получите ответ
- Или скажите **"Окей кафедра"** → говорите → получите ответ

### 4. Автозапуск (systemd)
```bash
sudo systemctl enable voice-assistant
sudo systemctl start voice-assistant
journalctl -u voice-assistant -f
```

## Структура проекта

```
src/
├── main.py               # Точка входа, оркестрация pipeline
├── config.py              # Загрузка YAML-конфигурации
├── asr/
│   ├── recognizer.py      # GigaAM v3 CTC ONNX (речь → текст)
│   └── wake_word.py       # Детекция wake word (Vosk)
├── rag/
│   ├── document_loader.py # Парсинг PDF/DOCX/TXT + chunking
│   ├── embedder.py        # multilingual-e5-base ONNX embeddings
│   ├── indexer.py         # FAISS + SQLite индексация
│   ├── retriever.py       # Семантический поиск
│   ├── generator.py       # Генерация ответа (template)
│   └── watcher.py         # Автоиндексация при изменении документов
├── tts/
│   └── synthesizer.py     # Piper TTS (текст → голос)
├── audio/
│   ├── recorder.py        # Запись с микрофона + VAD
│   └── player.py          # Воспроизведение аудио
├── hardware/
│   └── button.py          # GPIO-кнопка (+ клавиатурный fallback)
└── utils/
    ├── memory.py           # Управление RAM
    └── sounds.py           # Генерация системных звуков

config/assistant.yaml      # Конфигурация
data/documents/            # Документы базы знаний
data/models/               # Модели (не в git, скачиваются отдельно)
data/index/                # FAISS индекс + SQLite (генерируется)
benchmarks/                # Бенчмарки моделей
```

## Управление базой знаний

Положите файлы (PDF, DOCX, TXT) в `data/documents/`:

```bash
cp расписание.pdf data/documents/
```

**Автоиндексация**: система проверяет папку каждые 60 секунд и автоматически индексирует новые/изменённые файлы.

**Ручная индексация**:
```bash
bash scripts/index_documents.sh
```

## Оптимизация ONNX

Все модели сконвертированы в ONNX формат для максимальной производительности на CPU:

| Модель | PyTorch | ONNX | Ускорение |
|--------|---------|------|-----------|
| GigaAM v3 CTC (ASR) | 2.6 сек | 0.3 сек | **8.7x** |
| multilingual-e5-base (embed) | 4.7 сек | 1.8 сек | **2.6x** |
| Piper TTS | 1.5 сек | 1.2 сек | **1.3x** |

## Лицензии используемых компонентов

| Компонент | Лицензия |
|-----------|---------|
| GigaAM v3 | MIT |
| Piper TTS | MIT |
| FAISS | MIT |
| multilingual-e5-base | MIT |
| Vosk | Apache 2.0 |
