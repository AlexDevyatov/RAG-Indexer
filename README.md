# RAG Indexer

Локальный пайплайн индексации документов с веб-интерфейсом для RAG (Retrieval-Augmented Generation).

## Быстрый старт

```bash
# Установка и запуск
./deploy.sh --install-all

# Или вручную
pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull llama3.2:1b
python app.py
```

Открыть: http://localhost:8001

---

## Стек и архитектура (для LLM)

```
PROJECT: RAG Indexer
TYPE: Local RAG pipeline with web UI
LANGUAGE: Python 3.9+
FRAMEWORK: Flask

ARCHITECTURE:
┌─────────────────────────────────────────────────────────────────┐
│                        WEB UI (Flask)                           │
│                     http://localhost:8001                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  Drag&Drop   │  │   Search     │  │    Statistics        │  │
│  │   Upload     │  │    + Ask     │  │    Dashboard         │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BACKEND (app.py)                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ /api/upload  │  │  /api/ask    │  │  /api/search         │  │
│  │ /api/stats   │  │  /api/clear  │  │                      │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐
│     PARSER      │  │    CHUNKER      │  │     EMBEDDER        │
│  utils/parser.py│  │ utils/chunker.py│  │  utils/embedder.py  │
│                 │  │                 │  │                     │
│  - PDF (PyMuPDF)│  │  - Recursive    │  │  - Ollama API       │
│  - DOCX         │  │    splitting    │  │  - nomic-embed-text │
│  - TXT/MD       │  │  - 512 tokens   │  │  - 768 dimensions   │
│  - Code files   │  │  - 50 overlap   │  │  - Batch processing │
└─────────────────┘  └─────────────────┘  └─────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    VECTOR STORE (FAISS)                         │
│                                                                 │
│  ┌─────────────────────┐    ┌────────────────────────────────┐ │
│  │   faiss.index       │    │      metadata.json             │ │
│  │   IndexFlatL2       │    │   [{id, text, source, ...}]    │ │
│  │   L2 distance       │    │                                │ │
│  └─────────────────────┘    └────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM (DeepSeek API)                           │
│              https://api.deepseek.com/v1/chat                   │
│                                                                 │
│  Model: deepseek-chat                                           │
│  Task: Answer generation with context                           │
│  Prompt: System + Context chunks + User question                │
└─────────────────────────────────────────────────────────────────┘

DATA FLOW:
1. Upload: Files → Parser → Chunker → Embedder → FAISS Index
2. Query:  Question → Embedder → FAISS Search → Top-K chunks → LLM → Answer

TECH STACK:
- Backend:    Flask 3.x, Flask-CORS
- Vector DB:  FAISS (faiss-cpu), IndexFlatL2
- Embeddings: Ollama + nomic-embed-text (768 dim)
- LLM:        DeepSeek API (deepseek-chat)
- Parsing:    PyMuPDF (PDF), python-docx (DOCX)
- Frontend:   Vanilla JS, CSS Grid, Drag&Drop API

FILE STRUCTURE:
/
├── app.py              # Flask web server, API endpoints
├── index.py            # CLI indexing script
├── query.py            # CLI search script
├── deploy.sh           # Server deployment script
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Web UI with drag&drop
├── utils/
│   ├── parser.py       # Document parsing (PDF, DOCX, TXT, code)
│   ├── chunker.py      # Recursive text chunking
│   └── embedder.py     # Ollama embeddings wrapper
├── uploads/            # Uploaded files (gitignored)
├── index_data/         # FAISS index + metadata (gitignored)
│   ├── faiss.index
│   └── metadata.json
└── docs/               # Sample documents for testing

API ENDPOINTS:
POST /api/upload     - Upload and index files (multipart/form-data)
POST /api/ask        - RAG query: search + LLM answer
POST /api/search     - Vector search only (no LLM)
GET  /api/stats      - Index statistics
POST /api/clear      - Clear index and uploads

CONFIGURATION:
- Port: 8001 (configurable in app.py)
- Chunk size: 512 tokens
- Chunk overlap: 50 tokens
- Top-K results: 3
- LLM timeout: 60s
- Embedding model: Ollama nomic-embed-text
- Generation model: DeepSeek deepseek-chat

DEPENDENCIES (requirements.txt):
faiss-cpu>=1.7.4
PyMuPDF>=1.23.0
python-docx>=1.1.0
requests>=2.31.0
python-dotenv>=1.0.0
tqdm>=4.66.0
ollama>=0.1.0
flask>=3.0.0
flask-cors>=4.0.0
gunicorn>=21.0.0

MODELS REQUIRED:
- Ollama nomic-embed-text (274 MB) - for embeddings
- DeepSeek API key (.env) - for generation
```

---

## Использование

### Веб-интерфейс

1. Откройте http://localhost:8001
2. Перетащите документы в зону загрузки
3. Задайте вопрос в поле поиска

### CLI

```bash
# Индексация документов
python index.py --input docs/ --output my_index

# Поиск
python query.py --index my_index --query "Как установить?"

# Интерактивный режим
python query.py --index my_index --interactive
```

### API

```bash
# Загрузка файлов
curl -X POST http://localhost:8001/api/upload -F "files=@doc.pdf"

# RAG запрос
curl -X POST http://localhost:8001/api/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Как установить проект?", "top_k": 3}'

# Статистика
curl http://localhost:8001/api/stats
```

---

## Развёртывание на сервере

```bash
# Полная установка
./deploy.sh --install-all

# Управление
./start.sh       # Запуск
./stop.sh        # Остановка
./healthcheck.sh # Проверка

# Systemd (Linux)
sudo systemctl start rag-indexer
sudo systemctl status rag-indexer
```

Подробнее: [DEPLOY.md](DEPLOY.md)

---

## Лицензия

MIT
