#!/usr/bin/env python3
"""
Скрипт индексации документов для RAG.
Парсит документы, создаёт чанки, генерирует эмбеддинги и сохраняет FAISS индекс.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np

from utils.parser import parse_file, parse_directory, get_file_type
from utils.chunker import chunk_document
from utils.embedder import DeepSeekEmbedder


def create_faiss_index(embeddings: np.ndarray):
    """
    Создание FAISS индекса.
    
    Args:
        embeddings: numpy массив эмбеддингов [N, D]
        
    Returns:
        FAISS индекс
    """
    try:
        import faiss
    except ImportError:
        raise ImportError("📦 Установите FAISS: pip install faiss-cpu")
    
    # Получаем размерность эмбеддингов
    dimension = embeddings.shape[1]
    
    # Создаём индекс L2 (евклидово расстояние)
    index = faiss.IndexFlatL2(dimension)
    
    # Добавляем векторы
    index.add(embeddings.astype(np.float32))
    
    return index


def save_index(index, metadata: List[Dict], output_dir: str):
    """
    Сохранение индекса и метаданных.
    
    Args:
        index: FAISS индекс
        metadata: список метаданных чанков
        output_dir: директория для сохранения
    """
    try:
        import faiss
    except ImportError:
        raise ImportError("📦 Установите FAISS: pip install faiss-cpu")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем FAISS индекс
    index_path = output_path / "faiss.index"
    faiss.write_index(index, str(index_path))
    print(f"💾 Индекс сохранён: {index_path}")
    
    # Сохраняем метаданные
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"💾 Метаданные сохранены: {metadata_path}")


def index_documents(
    input_path: str,
    output_dir: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    batch_size: int = 10
):
    """
    Основная функция индексации документов.
    
    Args:
        input_path: путь к файлу или директории с документами
        output_dir: директория для сохранения индекса
        chunk_size: размер чанка в токенах
        chunk_overlap: размер перекрытия в токенах
        batch_size: размер батча для эмбеддингов
    """
    print("=" * 60)
    print("🚀 ИНДЕКСАЦИЯ ДОКУМЕНТОВ ДЛЯ RAG")
    print("=" * 60)
    
    # 1. Парсинг документов
    print(f"\n📂 Источник: {input_path}")
    
    input_path_obj = Path(input_path)
    
    if input_path_obj.is_file():
        # Один файл
        text, file_type = parse_file(str(input_path_obj))
        documents = [(str(input_path_obj), text, file_type)]
        print(f"📄 Обрабатывается файл: {input_path_obj.name}")
    elif input_path_obj.is_dir():
        # Директория
        documents = parse_directory(str(input_path_obj))
        print(f"📁 Найдено документов: {len(documents)}")
    else:
        raise FileNotFoundError(f"❌ Путь не существует: {input_path}")
    
    if not documents:
        print("⚠️ Не найдено документов для индексации!")
        return
    
    # 2. Чанкинг
    print(f"\n✂️ Чанкинг (размер: {chunk_size}, перекрытие: {chunk_overlap})")
    
    all_chunks = []
    for file_path, text, file_type in documents:
        chunks = chunk_document(text, file_path, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)
        print(f"  📄 {Path(file_path).name}: {len(chunks)} чанков")
    
    print(f"\n📦 Всего чанков: {len(all_chunks)}")
    
    if not all_chunks:
        print("⚠️ Не создано чанков для индексации!")
        return
    
    # 3. Генерация эмбеддингов
    print(f"\n🧠 Генерация эмбеддингов (батч: {batch_size})")
    
    embedder = DeepSeekEmbedder()
    texts = [chunk["text"] for chunk in all_chunks]
    
    embeddings = embedder.embed_texts(texts, batch_size=batch_size, show_progress=True)
    embeddings_array = np.array(embeddings)
    
    print(f"✅ Сгенерировано эмбеддингов: {len(embeddings)}")
    print(f"📐 Размерность: {embeddings_array.shape[1]}")
    
    # 4. Создание FAISS индекса
    print(f"\n🔧 Создание FAISS индекса")
    
    index = create_faiss_index(embeddings_array)
    print(f"✅ Индекс создан, всего векторов: {index.ntotal}")
    
    # 5. Сохранение
    print(f"\n💾 Сохранение в: {output_dir}")
    
    # Подготавливаем метаданные (без текста эмбеддингов - они в индексе)
    metadata = []
    for i, chunk in enumerate(all_chunks):
        metadata.append({
            "id": i,
            "text": chunk["text"],
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"],
            "total_chunks": chunk["total_chunks"]
        })
    
    save_index(index, metadata, output_dir)
    
    # Итоговая статистика
    print("\n" + "=" * 60)
    print("✅ ИНДЕКСАЦИЯ ЗАВЕРШЕНА")
    print("=" * 60)
    print(f"📁 Документов обработано: {len(documents)}")
    print(f"📦 Чанков создано: {len(all_chunks)}")
    print(f"🧠 Эмбеддингов сгенерировано: {len(embeddings)}")
    print(f"💾 Индекс сохранён в: {output_dir}")
    print("=" * 60)


def main():
    """Точка входа с парсингом аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Индексация документов для RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python index.py --input docs/ --output my_index
  python index.py --input document.pdf --output my_index
  python index.py --input src/ --output code_index --chunk-size 256
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Путь к файлу или директории с документами"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="my_index",
        help="Директория для сохранения индекса (по умолчанию: my_index)"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Размер чанка в токенах (по умолчанию: 512)"
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Размер перекрытия в токенах (по умолчанию: 50)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Размер батча для эмбеддингов (по умолчанию: 10)"
    )
    
    args = parser.parse_args()
    
    try:
        index_documents(
            input_path=args.input,
            output_dir=args.output,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            batch_size=args.batch_size
        )
    except KeyboardInterrupt:
        print("\n\n⚠️ Прервано пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
