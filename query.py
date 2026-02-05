#!/usr/bin/env python3
"""
Скрипт поиска и генерации ответов через RAG.
Выполняет семантический поиск по индексу и генерирует ответ через Ollama.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import requests

from utils.embedder import DeepSeekEmbedder


# Конфигурация Ollama
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2"


def load_index(index_dir: str):
    """
    Загрузка FAISS индекса и метаданных.
    
    Args:
        index_dir: директория с индексом
        
    Returns:
        кортеж (faiss_index, metadata)
    """
    try:
        import faiss
    except ImportError:
        raise ImportError("📦 Установите FAISS: pip install faiss-cpu")
    
    index_path = Path(index_dir)
    
    # Загружаем FAISS индекс
    faiss_file = index_path / "faiss.index"
    if not faiss_file.exists():
        raise FileNotFoundError(f"❌ Индекс не найден: {faiss_file}")
    
    index = faiss.read_index(str(faiss_file))
    print(f"📂 Загружен индекс: {index.ntotal} векторов")
    
    # Загружаем метаданные
    metadata_file = index_path / "metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"❌ Метаданные не найдены: {metadata_file}")
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"📋 Загружено метаданных: {len(metadata)} записей")
    
    return index, metadata


def search_index(
    query: str,
    index,
    metadata: List[Dict],
    embedder: DeepSeekEmbedder,
    top_k: int = 3
) -> List[Dict]:
    """
    Семантический поиск по индексу.
    
    Args:
        query: поисковый запрос
        index: FAISS индекс
        metadata: метаданные чанков
        embedder: эмбеддер
        top_k: количество результатов
        
    Returns:
        список релевантных чанков с дистанциями
    """
    # Генерируем эмбеддинг запроса
    print(f"🔍 Поиск: \"{query}\"")
    query_embedding = embedder.embed_single(query)
    query_vector = np.array([query_embedding]).astype(np.float32)
    
    # Поиск в индексе
    distances, indices = index.search(query_vector, top_k)
    
    # Формируем результаты
    results = []
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < len(metadata):
            result = metadata[idx].copy()
            result["distance"] = float(dist)
            result["rank"] = i + 1
            results.append(result)
    
    return results


def generate_response_ollama(
    query: str,
    context_chunks: List[Dict],
    model: str = OLLAMA_MODEL,
    stream: bool = True
) -> str:
    """
    Генерация ответа через Ollama с использованием контекста.
    
    Args:
        query: запрос пользователя
        context_chunks: релевантные чанки из индекса
        model: модель Ollama
        stream: потоковый вывод
        
    Returns:
        сгенерированный ответ
    """
    # Формируем контекст
    context_parts = []
    for chunk in context_chunks:
        source = Path(chunk["source"]).name
        context_parts.append(f"[Источник: {source}]\n{chunk['text']}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    # Формируем промпт
    system_prompt = """Ты — полезный ассистент, который отвечает на вопросы, используя предоставленный контекст.

Правила:
1. Отвечай ТОЛЬКО на основе предоставленного контекста
2. Если информации в контексте недостаточно, честно скажи об этом
3. Указывай источники информации, если это уместно
4. Отвечай на русском языке
5. Будь точным и конкретным"""

    user_prompt = f"""Контекст:
{context}

---

Вопрос: {query}

Ответ:"""

    # Запрос к Ollama
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": stream
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, stream=stream, timeout=120)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"❌ Не удалось подключиться к Ollama ({OLLAMA_URL})\n"
            "Убедитесь, что Ollama запущена: ollama serve"
        )
    except requests.exceptions.Timeout:
        raise TimeoutError("❌ Таймаут при запросе к Ollama")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"❌ Ошибка Ollama: {e}")
    
    # Обрабатываем ответ
    if stream:
        full_response = ""
        print("\n🤖 Ответ:\n")
        
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        chunk = data["message"]["content"]
                        full_response += chunk
                        print(chunk, end="", flush=True)
                    
                    if data.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue
        
        print("\n")
        return full_response
    else:
        data = response.json()
        return data["message"]["content"]


def query_rag(
    query: str,
    index_dir: str,
    top_k: int = 3,
    model: str = OLLAMA_MODEL,
    show_sources: bool = True
):
    """
    Основная функция RAG запроса.
    
    Args:
        query: запрос пользователя
        index_dir: директория с индексом
        top_k: количество чанков для контекста
        model: модель Ollama
        show_sources: показывать источники
    """
    print("=" * 60)
    print("🔎 RAG ЗАПРОС")
    print("=" * 60)
    
    # 1. Загружаем индекс
    print(f"\n📂 Загрузка индекса из: {index_dir}")
    index, metadata = load_index(index_dir)
    
    # 2. Инициализируем эмбеддер
    embedder = DeepSeekEmbedder()
    
    # 3. Поиск релевантных чанков
    print(f"\n🔍 Поиск топ-{top_k} релевантных фрагментов...")
    results = search_index(query, index, metadata, embedder, top_k)
    
    if not results:
        print("⚠️ Не найдено релевантных фрагментов!")
        return
    
    # 4. Показываем источники
    if show_sources:
        print(f"\n📚 Найденные источники:")
        for result in results:
            source = Path(result["source"]).name
            dist = result["distance"]
            text_preview = result["text"][:100].replace("\n", " ")
            print(f"  #{result['rank']} [{source}] (dist: {dist:.4f})")
            print(f"      \"{text_preview}...\"")
    
    # 5. Генерация ответа
    print(f"\n🧠 Генерация ответа через {model}...")
    
    response = generate_response_ollama(query, results, model)
    
    # Итог
    print("=" * 60)
    print("✅ ЗАПРОС ВЫПОЛНЕН")
    print("=" * 60)
    
    return response


def interactive_mode(index_dir: str, top_k: int = 3, model: str = OLLAMA_MODEL):
    """
    Интерактивный режим для множественных запросов.
    
    Args:
        index_dir: директория с индексом
        top_k: количество чанков для контекста
        model: модель Ollama
    """
    print("=" * 60)
    print("🎯 ИНТЕРАКТИВНЫЙ РЕЖИМ RAG")
    print("=" * 60)
    print("Введите 'выход' или 'exit' для завершения\n")
    
    # Загружаем индекс один раз
    index, metadata = load_index(index_dir)
    embedder = DeepSeekEmbedder()
    
    while True:
        try:
            query = input("\n❓ Ваш вопрос: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ("выход", "exit", "quit", "q"):
                print("\n👋 До свидания!")
                break
            
            # Поиск
            results = search_index(query, index, metadata, embedder, top_k)
            
            if not results:
                print("⚠️ Не найдено релевантных фрагментов!")
                continue
            
            # Генерация
            generate_response_ollama(query, results, model)
            
        except KeyboardInterrupt:
            print("\n\n👋 До свидания!")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")


def main():
    """Точка входа с парсингом аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Поиск и генерация ответов через RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python query.py --index my_index --query "Как установить проект?"
  python query.py --index my_index --query "Опиши архитектуру" --top-k 5
  python query.py --index my_index --interactive
        """
    )
    
    parser.add_argument(
        "--index", "-i",
        required=True,
        help="Директория с индексом"
    )
    
    parser.add_argument(
        "--query", "-q",
        help="Поисковый запрос"
    )
    
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=3,
        help="Количество релевантных чанков (по умолчанию: 3)"
    )
    
    parser.add_argument(
        "--model", "-m",
        default=OLLAMA_MODEL,
        help=f"Модель Ollama (по умолчанию: {OLLAMA_MODEL})"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Интерактивный режим"
    )
    
    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Не показывать источники"
    )
    
    args = parser.parse_args()
    
    try:
        if args.interactive:
            interactive_mode(args.index, args.top_k, args.model)
        elif args.query:
            query_rag(
                query=args.query,
                index_dir=args.index,
                top_k=args.top_k,
                model=args.model,
                show_sources=not args.no_sources
            )
        else:
            parser.error("Укажите --query или --interactive")
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Прервано пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
