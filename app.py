#!/usr/bin/env python3
"""
Веб-сервер для RAG системы.
Предоставляет API и веб-интерфейс для загрузки документов и поиска.
"""

import os
import json
import uuid
import shutil
from pathlib import Path
from typing import List, Optional

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np

from utils.parser import parse_file, get_file_type
from utils.chunker import chunk_document
from utils.embedder import OllamaEmbedder

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
    
    # Генерируем эмбеддинги
    embeddings = embedder.embed_texts(texts, show_progress=False)
    embeddings_array = np.array(embeddings).astype(np.float32)
    
    dimension = embeddings_array.shape[1]
    
    # Если индекс не существует - создаём
    if faiss_index is None:
        faiss_index = faiss.IndexFlatL2(dimension)
    
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
    """Генерация ответа через Ollama."""
    import requests
    
    # Формируем контекст
    context_parts = []
    for chunk in context_chunks:
        context_parts.append(f"[{chunk['filename']}]\n{chunk['text']}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    system_prompt = """Ты — полезный ассистент, который отвечает на вопросы на основе предоставленного контекста.
Отвечай точно и по делу. Если информации недостаточно, скажи об этом. Отвечай на русском языке."""

    user_prompt = f"""Контекст:
{context}

---

Вопрос: {query}"""

    payload = {
        "model": "llama3.2",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False
    }
    
    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json=payload,
            timeout=300
        )
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]
    except requests.exceptions.ConnectionError:
        return "❌ Ошибка: Ollama не запущена. Запустите: ollama serve"
    except requests.exceptions.Timeout:
        return "❌ Ошибка: Таймаут генерации. Попробуйте ещё раз."
    except Exception as e:
        return f"❌ Ошибка генерации: {str(e)}"


# === API ЭНДПОИНТЫ ===

@app.route('/')
def index():
    """Главная страница."""
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Загрузка и индексация файлов."""
    if 'files' not in request.files:
        return jsonify({"error": "Файлы не найдены"}), 400
    
    files = request.files.getlist('files')
    
    if not files or all(f.filename == '' for f in files):
        return jsonify({"error": "Файлы не выбраны"}), 400
    
    # Создаём папку для загрузок
    upload_path = Path(UPLOAD_FOLDER)
    upload_path.mkdir(parents=True, exist_ok=True)
    
    processed_files = []
    all_chunks = []
    all_sources = []
    
    for file in files:
        if file and file.filename and allowed_file(file.filename):
            # Сохраняем файл
            filename = secure_filename(file.filename)
            unique_name = f"{uuid.uuid4().hex[:8]}_{filename}"
            filepath = upload_path / unique_name
            file.save(str(filepath))
            
            try:
                # Парсим файл
                text, file_type = parse_file(str(filepath))
                
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
    
    # Индексируем все чанки
    if all_chunks:
        try:
            indexed_count = create_or_update_index(all_chunks, all_sources)
            return jsonify({
                "success": True,
                "files": processed_files,
                "indexed_chunks": indexed_count,
                "total_vectors": faiss_index.ntotal if faiss_index else 0
            })
        except Exception as e:
            return jsonify({
                "error": f"Ошибка индексации: {str(e)}",
                "files": processed_files
            }), 500
    
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
    
    try:
        # Поиск релевантных чанков
        results = search_index(query, top_k)
        
        if not results:
            return jsonify({
                "success": True,
                "query": query,
                "answer": "К сожалению, в индексе нет релевантной информации. Загрузите документы.",
                "sources": []
            })
        
        # Генерация ответа
        answer = generate_answer(query, results)
        
        return jsonify({
            "success": True,
            "query": query,
            "answer": answer,
            "sources": [{"filename": r["filename"], "distance": r["distance"]} for r in results]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
        
        return jsonify({"success": True, "message": "Индекс очищен"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Создаём необходимые директории
    Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(INDEX_FOLDER).mkdir(parents=True, exist_ok=True)
    Path('templates').mkdir(parents=True, exist_ok=True)
    Path('static').mkdir(parents=True, exist_ok=True)
    
    # Загружаем существующий индекс
    load_index()
    
    print("=" * 60)
    print("🚀 RAG WEB SERVER")
    print("=" * 60)
    print("📍 URL: http://localhost:8001")
    print("📂 Загрузки: uploads/")
    print("🗂️  Индекс: index_data/")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=8001, debug=False)
