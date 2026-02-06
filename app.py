#!/usr/bin/env python3
"""
Веб-сервер для RAG системы.
Предоставляет API и веб-интерфейс для загрузки документов и поиска.
"""

import os
import json
import uuid
import shutil
import time
import logging
from pathlib import Path
from typing import List, Optional
from threading import Lock
from datetime import datetime

from flask import Flask, request, jsonify, render_template, send_from_directory, Response, stream_with_context
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np

from utils.parser import parse_file, get_file_type
from utils.chunker import chunk_document
from utils.embedder import OllamaEmbedder

# === НАСТРОЙКА ЛОГИРОВАНИЯ ===
LOG_FOLDER = "logs"
Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)

# Формат логов
log_format = logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Логгер приложения
logger = logging.getLogger('rag_indexer')
logger.setLevel(logging.DEBUG)

# Файловый обработчик
file_handler = logging.FileHandler(
    f'{LOG_FOLDER}/app.log',
    encoding='utf-8'
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(log_format)

# Консольный обработчик
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(log_format)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Конфигурация
UPLOAD_FOLDER = "uploads"
INDEX_FOLDER = "index_data"
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'md', 'py', 'js', 'ts', 'java', 'cpp', 'c', 'go', 'rs', 'json', 'yaml', 'yml'}

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB макс
CORS(app)

# Глобальные переменные для индекса
faiss_index = None
metadata = []
embedder = None

# === ПРОГРЕСС ИНДЕКСАЦИИ ===
# Этапы: documents, parsing, chunking, embedding, faiss
# Статусы: pending, processing, completed, error
indexing_status = {
    "active": False,
    "steps": {
        "documents": {"status": "pending", "message": ""},
        "parsing": {"status": "pending", "message": ""},
        "chunking": {"status": "pending", "message": ""},
        "embedding": {"status": "pending", "message": ""},
        "faiss": {"status": "pending", "message": ""}
    },
    "error": None,
    "current_file": "",
    "total_files": 0,
    "processed_files": 0
}
indexing_lock = Lock()


def reset_indexing_status():
    """Сброс статуса индексации."""
    global indexing_status
    with indexing_lock:
        indexing_status = {
            "active": False,
            "steps": {
                "documents": {"status": "pending", "message": ""},
                "parsing": {"status": "pending", "message": ""},
                "chunking": {"status": "pending", "message": ""},
                "embedding": {"status": "pending", "message": ""},
                "faiss": {"status": "pending", "message": ""}
            },
            "error": None,
            "current_file": "",
            "total_files": 0,
            "processed_files": 0
        }


def update_step_status(step: str, status: str, message: str = ""):
    """Обновление статуса этапа."""
    global indexing_status
    with indexing_lock:
        if step in indexing_status["steps"]:
            indexing_status["steps"][step]["status"] = status
            indexing_status["steps"][step]["message"] = message


def set_indexing_error(error_message: str, failed_step: str = None):
    """Установка ошибки индексации."""
    global indexing_status
    with indexing_lock:
        indexing_status["error"] = error_message
        indexing_status["active"] = False
        if failed_step and failed_step in indexing_status["steps"]:
            indexing_status["steps"][failed_step]["status"] = "error"
            indexing_status["steps"][failed_step]["message"] = error_message


def allowed_file(filename: str) -> bool:
    """Проверка допустимого расширения файла."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def init_embedder():
    """Инициализация эмбеддера."""
    global embedder
    if embedder is None:
        try:
            embedder = OllamaEmbedder()
            print("✅ Эмбеддер инициализирован")
        except Exception as e:
            print(f"❌ Ошибка инициализации эмбеддера: {e}")
            raise


def load_index():
    """Загрузка существующего индекса."""
    global faiss_index, metadata
    
    try:
        import faiss
    except ImportError:
        print("❌ FAISS не установлен")
        return False
    
    index_path = Path(INDEX_FOLDER)
    faiss_file = index_path / "faiss.index"
    metadata_file = index_path / "metadata.json"
    
    if faiss_file.exists() and metadata_file.exists():
        faiss_index = faiss.read_index(str(faiss_file))
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print(f"📂 Загружен индекс: {faiss_index.ntotal} векторов")
        return True
    
    return False


def save_index():
    """Сохранение индекса на диск."""
    global faiss_index, metadata
    
    try:
        import faiss
    except ImportError:
        return False
    
    index_path = Path(INDEX_FOLDER)
    index_path.mkdir(parents=True, exist_ok=True)
    
    if faiss_index is not None:
        faiss.write_index(faiss_index, str(index_path / "faiss.index"))
        with open(index_path / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return True
    
    return False


def create_or_update_index(texts: List[str], sources: List[dict]):
    """Создание или обновление FAISS индекса."""
    global faiss_index, metadata, embedder
    
    try:
        import faiss
    except ImportError:
        raise ImportError("FAISS не установлен")
    
    init_embedder()
    
    # Обновляем статус - генерация эмбеддингов
    update_step_status("embedding", "processing", f"Генерация эмбеддингов для {len(texts)} чанков...")
    
    logger.info(f"🧠 Генерация эмбеддингов для {len(texts)} чанков...")
    embed_start = time.time()
    
    # Генерируем эмбеддинги
    embeddings = embedder.embed_texts(texts, show_progress=False)
    embeddings_array = np.array(embeddings).astype(np.float32)
    
    embed_time = time.time() - embed_start
    logger.info(f"⏱️  Эмбеддинги сгенерированы за {embed_time:.2f} сек ({len(texts)/embed_time:.1f} чанков/сек)")
    
    update_step_status("embedding", "completed", f"Сгенерировано {len(texts)} эмбеддингов")
    
    # Обновляем статус - сохранение в FAISS
    update_step_status("faiss", "processing", "Сохранение в FAISS индекс...")
    
    dimension = embeddings_array.shape[1]
    logger.debug(f"Размерность эмбеддингов: {dimension}")
    
    # Если индекс не существует - создаём
    if faiss_index is None:
        faiss_index = faiss.IndexFlatL2(dimension)
        logger.info(f"📦 Создан новый FAISS индекс (dim={dimension})")
    
    # Добавляем новые векторы
    start_id = faiss_index.ntotal
    faiss_index.add(embeddings_array)
    
    # Добавляем метаданные
    for i, (text, source) in enumerate(zip(texts, sources)):
        metadata.append({
            "id": start_id + i,
            "text": text,
            "source": source["source"],
            "filename": source["filename"],
            "chunk_index": source.get("chunk_index", 0)
        })
    
    # Сохраняем
    save_index()
    
    logger.info(f"💾 Индекс сохранён: {faiss_index.ntotal} векторов")
    update_step_status("faiss", "completed", f"Индекс содержит {faiss_index.ntotal} векторов")
    
    return len(texts)


def search_index(query: str, top_k: int = 3) -> List[dict]:
    """Поиск по индексу."""
    global faiss_index, metadata, embedder
    
    if faiss_index is None or faiss_index.ntotal == 0:
        return []
    
    init_embedder()
    
    # Получаем эмбеддинг запроса
    query_embedding = embedder.embed_single(query)
    query_vector = np.array([query_embedding]).astype(np.float32)
    
    # Поиск
    distances, indices = faiss_index.search(query_vector, min(top_k, faiss_index.ntotal))
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(metadata):
            result = metadata[idx].copy()
            result["distance"] = float(dist)
            results.append(result)
    
    return results


def generate_answer(query: str, context_chunks: List[dict]) -> str:
    """Генерация ответа через DeepSeek API."""
    import requests
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    
    if not api_key:
        logger.error("DEEPSEEK_API_KEY не найден в .env")
        return "❌ Ошибка: DEEPSEEK_API_KEY не найден в .env"
    
    # Формируем контекст
    context_parts = []
    for chunk in context_chunks:
        context_parts.append(f"[{chunk['filename']}]\n{chunk['text']}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    # Логируем размер контекста
    context_chars = len(context)
    context_words = len(context.split())
    logger.info(f"📊 Контекст: {len(context_chunks)} чанков, {context_chars} символов, ~{context_words} слов")
    logger.debug(f"Вопрос: {query[:100]}...")
    
    system_prompt = """Ты — полезный ассистент, который отвечает на вопросы на основе предоставленного контекста.
Отвечай точно и по делу. Если информации недостаточно, скажи об этом. Отвечай на русском языке."""

    user_prompt = f"""Контекст:
{context}

---

Вопрос: {query}"""

    # Оцениваем размер промпта (примерно 4 символа = 1 токен)
    estimated_tokens = (len(system_prompt) + len(user_prompt)) // 4
    logger.info(f"📝 Примерный размер промпта: ~{estimated_tokens} токенов")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 2048
    }
    
    try:
        logger.info("🚀 Отправка запроса к DeepSeek API...")
        start_time = time.time()
        
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120  # Увеличен таймаут до 120 секунд
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"⏱️  Время ответа DeepSeek API: {elapsed_time:.2f} сек")
        
        if response.status_code == 401:
            logger.error("Неверный API ключ DeepSeek")
            return "❌ Ошибка: Неверный API ключ DeepSeek"
        
        if response.status_code == 429:
            logger.error("Превышен лимит запросов DeepSeek API")
            return "❌ Ошибка: Превышен лимит запросов DeepSeek API"
        
        response.raise_for_status()
        data = response.json()
        
        # Логируем использование токенов
        if 'usage' in data:
            usage = data['usage']
            logger.info(f"📈 Токены: prompt={usage.get('prompt_tokens', '?')}, "
                       f"completion={usage.get('completion_tokens', '?')}, "
                       f"total={usage.get('total_tokens', '?')}")
        
        answer = data["choices"][0]["message"]["content"]
        logger.info(f"✅ Ответ получен: {len(answer)} символов")
        logger.debug(f"Ответ: {answer[:200]}...")
        
        return answer
        
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Ошибка подключения к DeepSeek API: {e}")
        return "❌ Ошибка: Не удалось подключиться к DeepSeek API"
    except requests.exceptions.Timeout:
        elapsed_time = time.time() - start_time
        logger.error(f"❌ ТАЙМАУТ DeepSeek API после {elapsed_time:.2f} сек! "
                    f"Контекст: {context_chars} символов, ~{estimated_tokens} токенов")
        return "❌ Превышено время ожидания ответа от модели. Попробуйте упростить запрос."
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP ошибка: {e.response.status_code} - {e.response.text}")
        return f"❌ Ошибка HTTP: {e.response.status_code} - {e.response.text}"
    except Exception as e:
        logger.exception(f"Неожиданная ошибка генерации: {e}")
        return f"❌ Ошибка генерации: {str(e)}"


# === API ЭНДПОИНТЫ ===

@app.route('/')
def index():
    """Главная страница."""
    return render_template('index.html')


@app.route('/api/progress')
def progress_stream():
    """SSE эндпоинт для отслеживания прогресса индексации."""
    def generate():
        while True:
            with indexing_lock:
                data = json.dumps(indexing_status)
            yield f"data: {data}\n\n"
            
            # Если индексация не активна и нет ошибки, проверяем реже
            with indexing_lock:
                is_active = indexing_status["active"]
            
            if is_active:
                time.sleep(0.5)  # Чаще проверяем во время индексации
            else:
                time.sleep(2)  # Реже проверяем в режиме ожидания
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )


@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Загрузка и индексация файлов."""
    global indexing_status
    
    if 'files' not in request.files:
        return jsonify({"error": "Файлы не найдены"}), 400
    
    files = request.files.getlist('files')
    
    if not files or all(f.filename == '' for f in files):
        return jsonify({"error": "Файлы не выбраны"}), 400
    
    # Сбрасываем и активируем статус индексации
    reset_indexing_status()
    with indexing_lock:
        indexing_status["active"] = True
        indexing_status["total_files"] = len([f for f in files if f and f.filename and allowed_file(f.filename)])
    
    # Этап 1: Документы
    update_step_status("documents", "processing", f"Получено {len(files)} файлов")
    
    # Создаём папку для загрузок
    upload_path = Path(UPLOAD_FOLDER)
    upload_path.mkdir(parents=True, exist_ok=True)
    
    processed_files = []
    all_chunks = []
    all_sources = []
    
    update_step_status("documents", "completed", f"Загружено {len(files)} файлов")
    
    # Этап 2: Парсинг
    update_step_status("parsing", "processing", "Извлечение текста из документов...")
    
    for file in files:
        if file and file.filename and allowed_file(file.filename):
            # Сохраняем файл
            filename = secure_filename(file.filename)
            unique_name = f"{uuid.uuid4().hex[:8]}_{filename}"
            filepath = upload_path / unique_name
            file.save(str(filepath))
            
            with indexing_lock:
                indexing_status["current_file"] = filename
            
            try:
                # Парсим файл
                text, file_type = parse_file(str(filepath))
                
                with indexing_lock:
                    indexing_status["processed_files"] += 1
                
                update_step_status("parsing", "processing", f"Обработан: {filename}")
                
                # Этап 3: Чанкинг (обновляется для каждого файла)
                update_step_status("chunking", "processing", f"Разбиение на чанки: {filename}")
                
                # Чанкинг
                chunks = chunk_document(text, str(filepath), chunk_size=512, chunk_overlap=50)
                
                for chunk in chunks:
                    all_chunks.append(chunk["text"])
                    all_sources.append({
                        "source": str(filepath),
                        "filename": filename,
                        "chunk_index": chunk["chunk_index"]
                    })
                
                processed_files.append({
                    "filename": filename,
                    "chunks": len(chunks),
                    "type": file_type
                })
                
            except Exception as e:
                processed_files.append({
                    "filename": filename,
                    "error": str(e)
                })
                set_indexing_error(f"Ошибка при обработке {filename}: {str(e)}", "parsing")
    
    update_step_status("parsing", "completed", f"Обработано {len(processed_files)} файлов")
    update_step_status("chunking", "completed", f"Создано {len(all_chunks)} чанков")
    
    # Индексируем все чанки
    if all_chunks:
        try:
            indexed_count = create_or_update_index(all_chunks, all_sources)
            
            # Завершаем индексацию
            with indexing_lock:
                indexing_status["active"] = False
                indexing_status["current_file"] = ""
            
            return jsonify({
                "success": True,
                "files": processed_files,
                "indexed_chunks": indexed_count,
                "total_vectors": faiss_index.ntotal if faiss_index else 0
            })
        except Exception as e:
            set_indexing_error(f"Ошибка индексации: {str(e)}", "embedding")
            return jsonify({
                "error": f"Ошибка индексации: {str(e)}",
                "files": processed_files
            }), 500
    
    # Завершаем без чанков
    with indexing_lock:
        indexing_status["active"] = False
    
    return jsonify({
        "success": True,
        "files": processed_files,
        "indexed_chunks": 0
    })


@app.route('/api/search', methods=['POST'])
def search():
    """Поиск по индексу."""
    data = request.get_json()
    
    if not data or 'query' not in data:
        return jsonify({"error": "Запрос не указан"}), 400
    
    query = data['query']
    top_k = data.get('top_k', 3)
    
    try:
        results = search_index(query, top_k)
        return jsonify({
            "success": True,
            "query": query,
            "results": results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/ask', methods=['POST'])
def ask():
    """RAG запрос: поиск + генерация ответа."""
    data = request.get_json()
    
    if not data or 'query' not in data:
        return jsonify({"error": "Вопрос не указан"}), 400
    
    query = data['query']
    top_k = data.get('top_k', 3)
    
    logger.info("=" * 50)
    logger.info(f"🔍 Новый RAG-запрос: '{query[:80]}...' (top_k={top_k})")
    request_start = time.time()
    
    try:
        # Поиск релевантных чанков
        search_start = time.time()
        results = search_index(query, top_k)
        search_time = time.time() - search_start
        logger.info(f"⏱️  Поиск в FAISS: {search_time:.3f} сек, найдено {len(results)} чанков")
        
        if not results:
            logger.warning("Релевантные документы не найдены")
            return jsonify({
                "success": True,
                "query": query,
                "answer": "К сожалению, в индексе нет релевантной информации. Загрузите документы.",
                "sources": []
            })
        
        # Логируем найденные источники
        for i, r in enumerate(results):
            logger.debug(f"  [{i+1}] {r['filename']} (distance: {r['distance']:.4f})")
        
        # Генерация ответа
        generation_start = time.time()
        answer = generate_answer(query, results)
        generation_time = time.time() - generation_start
        
        total_time = time.time() - request_start
        logger.info(f"⏱️  Общее время запроса: {total_time:.2f} сек "
                   f"(поиск: {search_time:.3f}с, генерация: {generation_time:.2f}с)")
        
        # Проверяем, содержит ли ответ ошибку таймаута
        if "Превышено время ожидания" in answer:
            logger.error(f"❌ Таймаут после {total_time:.2f} сек")
            return jsonify({
                "error": "Превышено время ожидания ответа от модели. Попробуйте упростить запрос."
            }), 504
        
        # Проверяем другие ошибки
        if answer.startswith("❌"):
            logger.error(f"Ошибка генерации: {answer}")
            return jsonify({
                "error": answer.replace("❌ ", "")
            }), 500
        
        logger.info(f"✅ Запрос успешно обработан за {total_time:.2f} сек")
        
        return jsonify({
            "success": True,
            "query": query,
            "answer": answer,
            "sources": [{"filename": r["filename"], "distance": r["distance"]} for r in results]
        })
        
    except requests.exceptions.Timeout:
        total_time = time.time() - request_start
        logger.error(f"❌ Таймаут запроса после {total_time:.2f} сек")
        return jsonify({
            "error": "Превышено время ожидания ответа от модели. Попробуйте упростить запрос."
        }), 504
    except Exception as e:
        logger.exception(f"Неожиданная ошибка в /api/ask: {e}")
        return jsonify({"error": f"Ошибка генерации: {str(e)}"}), 500


@app.route('/api/stats', methods=['GET'])
def stats():
    """Статистика индекса."""
    return jsonify({
        "total_vectors": faiss_index.ntotal if faiss_index else 0,
        "total_documents": len(set(m.get("filename", "") for m in metadata)) if metadata else 0,
        "total_chunks": len(metadata)
    })


@app.route('/api/clear', methods=['POST'])
def clear_index():
    """Очистка индекса."""
    global faiss_index, metadata
    
    try:
        import faiss
        faiss_index = None
        metadata = []
        
        # Удаляем файлы
        index_path = Path(INDEX_FOLDER)
        if index_path.exists():
            shutil.rmtree(index_path)
        
        upload_path = Path(UPLOAD_FOLDER)
        if upload_path.exists():
            shutil.rmtree(upload_path)
        
        # Сбрасываем статус индексации
        reset_indexing_status()
        
        return jsonify({"success": True, "message": "Индекс очищен"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Импортируем requests для использования в ask
    import requests
    
    # Создаём необходимые директории
    Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(INDEX_FOLDER).mkdir(parents=True, exist_ok=True)
    Path('templates').mkdir(parents=True, exist_ok=True)
    Path('static').mkdir(parents=True, exist_ok=True)
    
    # Загружаем существующий индекс
    load_index()
    
    logger.info("=" * 60)
    logger.info("🚀 RAG WEB SERVER ЗАПУЩЕН")
    logger.info("=" * 60)
    logger.info(f"📍 URL: http://localhost:8001")
    logger.info(f"📂 Загрузки: {UPLOAD_FOLDER}/")
    logger.info(f"🗂️  Индекс: {INDEX_FOLDER}/")
    logger.info(f"📋 Логи: {LOG_FOLDER}/app.log")
    logger.info("=" * 60)
    
    print("=" * 60)
    print("🚀 RAG WEB SERVER")
    print("=" * 60)
    print("📍 URL: http://localhost:8001")
    print("📂 Загрузки: uploads/")
    print("🗂️  Индекс: index_data/")
    print(f"📋 Логи: {LOG_FOLDER}/app.log")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=8001, debug=False)
