"""
Модуль рекурсивного чанкинга текста.
Разбивает текст на части с заданным размером и перекрытием.
"""

from typing import List, Optional


# Разделители для рекурсивного чанкинга (от более крупных к мелким)
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " "]


def count_tokens_approx(text: str) -> int:
    """
    Приблизительный подсчёт токенов (1 токен ≈ 4 символа для английского,
    для русского ≈ 2-3 символа).
    Используем среднее значение 3.5 символа на токен.
    
    Args:
        text: входной текст
        
    Returns:
        приблизительное количество токенов
    """
    return len(text) // 3


def split_by_separator(text: str, separator: str) -> List[str]:
    """
    Разделение текста по разделителю с сохранением разделителя.
    
    Args:
        text: входной текст
        separator: разделитель
        
    Returns:
        список частей текста
    """
    if not separator:
        return list(text)
    
    parts = text.split(separator)
    # Добавляем разделитель обратно к каждой части (кроме последней)
    result = []
    for i, part in enumerate(parts):
        if i < len(parts) - 1:
            result.append(part + separator)
        else:
            result.append(part)
    
    return [p for p in result if p]  # Убираем пустые строки


def recursive_chunk(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    separators: Optional[List[str]] = None
) -> List[str]:
    """
    Рекурсивный чанкинг текста.
    
    Алгоритм:
    1. Пытаемся разбить по текущему разделителю
    2. Если часть слишком большая - рекурсивно разбиваем по следующему разделителю
    3. Объединяем части до достижения целевого размера
    4. Добавляем перекрытие между чанками
    
    Args:
        text: входной текст
        chunk_size: целевой размер чанка в токенах
        chunk_overlap: размер перекрытия в токенах
        separators: список разделителей (от крупных к мелким)
        
    Returns:
        список текстовых чанков
    """
    if separators is None:
        separators = DEFAULT_SEPARATORS.copy()
    
    # Базовый случай: текст меньше целевого размера
    if count_tokens_approx(text) <= chunk_size:
        return [text.strip()] if text.strip() else []
    
    # Если разделителей не осталось - принудительное разбиение по символам
    if not separators:
        return force_split(text, chunk_size, chunk_overlap)
    
    # Берём текущий разделитель
    separator = separators[0]
    remaining_separators = separators[1:]
    
    # Разбиваем текст
    parts = split_by_separator(text, separator)
    
    chunks = []
    current_chunk = ""
    
    for part in parts:
        part_tokens = count_tokens_approx(part)
        current_tokens = count_tokens_approx(current_chunk)
        
        # Если часть сама по себе слишком большая - рекурсивно разбиваем
        if part_tokens > chunk_size:
            # Сначала сохраняем текущий чанк если есть
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # Рекурсивно разбиваем большую часть
            sub_chunks = recursive_chunk(
                part, chunk_size, chunk_overlap, remaining_separators
            )
            chunks.extend(sub_chunks)
            continue
        
        # Если добавление части превысит лимит - сохраняем текущий чанк
        if current_tokens + part_tokens > chunk_size and current_chunk.strip():
            chunks.append(current_chunk.strip())
            
            # Добавляем перекрытие
            overlap_text = get_overlap(current_chunk, chunk_overlap)
            current_chunk = overlap_text + part
        else:
            current_chunk += part
    
    # Добавляем последний чанк
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def force_split(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Принудительное разбиение текста по символам когда разделители не работают.
    
    Args:
        text: входной текст
        chunk_size: целевой размер в токенах
        chunk_overlap: размер перекрытия в токенах
        
    Returns:
        список чанков
    """
    # Конвертируем токены в символы (примерно)
    char_size = chunk_size * 3
    char_overlap = chunk_overlap * 3
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + char_size
        chunk = text[start:end].strip()
        
        if chunk:
            chunks.append(chunk)
        
        start = end - char_overlap
    
    return chunks


def get_overlap(text: str, overlap_tokens: int) -> str:
    """
    Получение текста для перекрытия (последние N токенов).
    
    Args:
        text: исходный текст
        overlap_tokens: количество токенов для перекрытия
        
    Returns:
        текст перекрытия
    """
    # Конвертируем токены в символы
    overlap_chars = overlap_tokens * 3
    
    if len(text) <= overlap_chars:
        return text
    
    return text[-overlap_chars:]


def chunk_document(
    text: str,
    source: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50
) -> List[dict]:
    """
    Чанкинг документа с добавлением метаданных.
    
    Args:
        text: текст документа
        source: путь к исходному файлу
        chunk_size: размер чанка в токенах
        chunk_overlap: размер перекрытия в токенах
        
    Returns:
        список словарей с чанками и метаданными
    """
    chunks = recursive_chunk(text, chunk_size, chunk_overlap)
    
    result = []
    for i, chunk in enumerate(chunks):
        result.append({
            "text": chunk,
            "source": source,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "token_count_approx": count_tokens_approx(chunk)
        })
    
    return result


if __name__ == "__main__":
    # Пример использования
    sample_text = """
    Python — высокоуровневый язык программирования общего назначения с динамической 
    строгой типизацией и автоматическим управлением памятью.
    
    Язык создан в конце 1980-х годов Гвидо ван Россумом, первая версия вышла в 1991 году.
    
    Основные особенности Python:
    - Простой и понятный синтаксис
    - Динамическая типизация
    - Автоматическое управление памятью
    - Богатая стандартная библиотека
    - Множество сторонних библиотек
    
    Python широко используется в:
    - Веб-разработке (Django, Flask)
    - Науке о данных и машинном обучении (NumPy, Pandas, TensorFlow)
    - Автоматизации и скриптинге
    - Разработке игр
    - Системном администрировании
    """
    
    print("🔧 Тестирование чанкера\n")
    print(f"📝 Исходный текст: {len(sample_text)} символов, ~{count_tokens_approx(sample_text)} токенов")
    
    chunks = chunk_document(sample_text, "test.txt", chunk_size=100, chunk_overlap=20)
    
    print(f"\n📦 Получено чанков: {len(chunks)}\n")
    
    for chunk in chunks:
        print(f"--- Чанк {chunk['chunk_index'] + 1}/{chunk['total_chunks']} ---")
        print(f"📊 ~{chunk['token_count_approx']} токенов")
        print(f"📄 Текст: {chunk['text'][:100]}...")
        print()
