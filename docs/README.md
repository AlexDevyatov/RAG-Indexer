# RAG Indexer - Система индексации документов

## Описание проекта

RAG Indexer — это локальный пайплайн для индексации и поиска по документам с использованием технологии RAG (Retrieval-Augmented Generation).

Система позволяет:
- Индексировать документы различных форматов (PDF, DOCX, TXT, MD, код)
- Выполнять семантический поиск по индексу
- Генерировать ответы на вопросы с использованием контекста из документов

## Установка проекта

### Требования

- Python 3.9 или выше
- Ollama (для локальной генерации ответов)
- DeepSeek API ключ (для эмбеддингов)

### Шаги установки

1. Клонируйте репозиторий:
```bash
git clone https://github.com/example/rag-indexer.git
cd rag-indexer
```

2. Создайте виртуальное окружение:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

4. Создайте файл `.env` и добавьте API ключ:
```
DEEPSEEK_API_KEY=sk-your-api-key-here
```

5. Установите и запустите Ollama:
```bash
brew install ollama  # macOS
ollama pull llama3.2
ollama serve
```

## Использование

### Индексация документов

```bash
python index.py --input docs/ --output my_index
```

Параметры:
- `--input` или `-i`: путь к файлу или директории
- `--output` или `-o`: директория для сохранения индекса
- `--chunk-size`: размер чанка в токенах (по умолчанию 512)
- `--chunk-overlap`: размер перекрытия (по умолчанию 50)

### Поиск и генерация ответов

```bash
python query.py --index my_index --query "Как установить проект?"
```

Параметры:
- `--index` или `-i`: директория с индексом
- `--query` или `-q`: поисковый запрос
- `--top-k`: количество релевантных чанков (по умолчанию 3)
- `--interactive`: интерактивный режим

## Архитектура

### Компоненты системы

1. **Parser** (`utils/parser.py`) — парсинг документов
2. **Chunker** (`utils/chunker.py`) — разбиение на чанки
3. **Embedder** (`utils/embedder.py`) — генерация эмбеддингов
4. **FAISS Index** — хранение и поиск векторов

### Поток данных

```
Документы → Парсинг → Чанкинг → Эмбеддинги → FAISS Index
                                                    ↓
Запрос → Эмбеддинг → Поиск → Контекст → LLM → Ответ
```

## Лицензия

MIT License
